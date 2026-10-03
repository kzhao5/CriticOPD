# Semantic EOS class for on-policy distillation.
#
# Motivation (paper "When EOS Tokens Disagree: Understanding Length Inflation in
# On-Policy Distillation"): teacher and student may put their stop mass on
# different terminal tokens (e.g. <|im_end|> vs <|endoftext|>) even with a shared
# vocabulary. Sampled-token OPD then scores the student's own EOS with
#   A = log p_T(eos_S) - log p_S(eos_S)
# which is strongly negative although *both* models want to stop, so the student
# is pushed away from stopping. The fix merges the task-equivalent terminal
# tokens E into ONE action STOP:
#   log p(STOP | prefix) = logsumexp_{e in E} log p(e | prefix)
# where log p(e | prefix) are full-vocabulary-normalised token log-probs.
#
# This module is backend-agnostic (torch only) so it can be unit tested on CPU
# and reused by the FSDP engine (student / old-policy log-probs) and by the loss.
from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)


def parse_eos_ids(spec) -> list[int]:
    """Normalise an EOS-set specification to a sorted list of unique ints.

    Accepts None / "" (-> []), an int, an iterable of ints, or a comma separated
    string such as "151645,151643" (the form used for env vars / batch meta).
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.replace(";", ",").split(",")]
        ids = [int(p) for p in parts if p]
    elif isinstance(spec, int):
        ids = [spec]
    else:
        ids = [int(x) for x in spec]
    return sorted(set(ids))


def stop_logprob_from_logits(logit_rows: torch.Tensor, eos_ids: torch.Tensor) -> torch.Tensor:
    """log p(STOP) for each row of logits, normalised over the FULL vocabulary.

    Args:
        logit_rows: (n, vocab) logits (any float dtype; math is done in float32).
        eos_ids: (|E|,) long tensor of terminal token ids.
    Returns:
        (n,) float32 tensor  logsumexp_{e in E} logits[e] - logsumexp_v logits[v].
    Gradient flows through both logsumexp terms (softmax_E - softmax_full).
    """
    lp = logit_rows.float()
    lse_full = torch.logsumexp(lp, dim=-1)
    lse_eos = torch.logsumexp(lp.index_select(-1, eos_ids.to(lp.device)), dim=-1)
    return lse_eos - lse_full


def semantic_eos_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_probs: torch.Tensor,
    eos_ids: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace log p(label) by log p(STOP) at rows whose label is a terminal token.

    Args:
        logits: (N, vocab) logits exactly as consumed by the cross-entropy that
            produced ``log_probs`` (temperature already applied). Only the rows
            whose label is in E are read (cheap: a few rows per batch).
        labels: (N,) next-token ids (the sampled actions).
        log_probs: (N,) log p(label) with autograd history (e.g. flash cross-entropy).
        eos_ids: terminal token ids that are equivalent for the task.
    Returns:
        new_log_probs: (N,) ``log_probs`` with EOS rows replaced by log p(STOP)
            (gradient kept through the logsumexp; other rows untouched).
        raw_eos_log_probs: (N,) zeros except EOS rows, where it holds the
            *original* sampled-token log-prob (detached) for diagnostics.
    Non-EOS rows are not renormalised: merging the EOS coordinates does not
    change any other token's probability.
    """
    eos_list = parse_eos_ids(eos_ids)
    raw = torch.zeros_like(log_probs)
    if not eos_list:
        return log_probs, raw
    if logits.dim() != 2 or labels.dim() != 1 or log_probs.dim() != 1:
        raise ValueError(
            f"semantic_eos_logprobs expects logits (N,V), labels (N,), log_probs (N,); "
            f"got {tuple(logits.shape)}, {tuple(labels.shape)}, {tuple(log_probs.shape)}"
        )
    eos_t = torch.tensor(eos_list, dtype=torch.long, device=labels.device)
    is_eos = torch.isin(labels, eos_t)
    if not bool(is_eos.any()):
        return log_probs, raw
    rows = torch.nonzero(is_eos, as_tuple=False).flatten()
    stop_lp = stop_logprob_from_logits(logits.index_select(0, rows), eos_t.to(logits.device))
    raw = raw.index_put((rows,), log_probs.detach()[rows])
    new_log_probs = log_probs.index_put((rows,), stop_lp.to(log_probs.dtype))
    return new_log_probs, raw


def merge_stop_distribution(log_probs_full: torch.Tensor, eos_ids: Iterable[int]) -> torch.Tensor:
    """Aggregate a full-vocabulary log-distribution into the semantic action space.

    Returns log-probs over (non-terminal tokens ..., STOP): all coordinates in E
    are removed and ONE STOP coordinate (appended last) carries their total mass.
    Use this to build the action distribution for full-vocabulary KL variants; the
    result is normalised (logsumexp == 0) whenever the input is.
    """
    eos_list = parse_eos_ids(eos_ids)
    if not eos_list:
        return log_probs_full
    vocab = log_probs_full.shape[-1]
    eos_t = torch.tensor(eos_list, dtype=torch.long, device=log_probs_full.device)
    keep = torch.ones(vocab, dtype=torch.bool, device=log_probs_full.device)
    keep[eos_t] = False
    non_eos = log_probs_full[..., keep]
    stop = torch.logsumexp(log_probs_full.index_select(-1, eos_t), dim=-1, keepdim=True)
    return torch.cat([non_eos, stop], dim=-1)


def resolve_semantic_eos_ids(
    explicit: Optional[Iterable[int]],
    student_path: Optional[str],
    teacher_path: Optional[str] = None,
) -> list[int]:
    """Resolve the terminal-token set E for a training run.

    Priority: explicit list (user-confirmed) > terminal tokens declared by the
    student's generation_config.json (``eos_token_id``) plus the tokenizer's
    ``eos_token_id``. The teacher's declared set is loaded only to WARN on
    mismatch; declared sets are *configuration*, not measured stop preference
    (see eos_fix/eos_diag.py for the measured probabilities).
    """
    ids = parse_eos_ids(explicit)
    if ids:
        logger.warning("[semantic_eos] using explicit terminal token set E=%s", ids)
        return ids
    if not student_path:
        raise ValueError("semantic_eos: need either explicit ids or a student model path")

    def _declared(path: str) -> list[int]:
        out: list[int] = []
        try:
            from transformers import GenerationConfig

            gc = GenerationConfig.from_pretrained(path)
            out += parse_eos_ids(gc.eos_token_id)
        except Exception as e:  # pragma: no cover - depends on files on disk
            logger.warning("[semantic_eos] no generation_config for %s (%s)", path, e)
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if tok.eos_token_id is not None:
                out += parse_eos_ids(tok.eos_token_id)
        except Exception as e:  # pragma: no cover
            logger.warning("[semantic_eos] no tokenizer for %s (%s)", path, e)
        return sorted(set(out))

    student_ids = _declared(student_path)
    if not student_ids:
        raise ValueError(f"semantic_eos: could not resolve any terminal token for student {student_path}")
    ids = list(student_ids)
    if teacher_path:
        teacher_ids = _declared(teacher_path)
        if teacher_ids and set(teacher_ids) != set(student_ids):
            # The paper's failure mode is exactly a student that declares/uses one terminal token while the
            # teacher prefers another (e.g. Qwen3-*-Base: <|endoftext|> only; post-trained Qwen3: <|im_end|>).
            # Default E = union of both declared single-turn terminal sets; pass semantic_eos_token_ids
            # explicitly if some of these tokens are NOT task-equivalent (multi-turn / tool end-of-turn).
            ids = sorted(set(student_ids) | set(teacher_ids))
            logger.warning(
                "[semantic_eos] student declares terminal tokens %s, teacher declares %s -> using the UNION E=%s",
                student_ids,
                teacher_ids,
                ids,
            )
    logger.warning("[semantic_eos] resolved terminal token set E=%s (student=%s)", ids, student_path)
    return ids
