# CriticOPD: the student writes a full solution -> it is graded wrong -> the teacher points out the first error and gives feedback -> the student rewrites with the feedback.
#
# Difference from Relay/Lucid: those two "hand off" during sampling (the teacher continues from the same prefix); this is
# post-hoc critique + rewrite, where the teacher generates on an entirely different prompt (the critic prompt) and does not solve the problem.
#
# Training-side convention (response_mask):
#   kept prefix  -> 0   feedback -> not in the sequence   student rewrite tokens -> 1
# That is, the feedback is in the context only during generation and is removed for training; the loss falls only on the repair tokens the student wrote itself.
# This is what "make the student internalize the correction" means: there is no teacher at test time, so the student must write the corrected part on its own.
import asyncio
import json
import random
import logging, os, re

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # relay-opd/
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SEP = re.compile(r"\n\s*\n")
_LEADIN = re.compile(r"^\s*(?:#{1,6}\s|[-*_=]{3,}\s*$|\*\*[^*]{0,40}\*\*\s*:?\s*$"
                     r"|(?:So|Now|Then|Next|Thus|Hence|Therefore|Step\s*\d+|Case\s*\d+)\b[^.\n]{0,30}:\s*$)", re.I)
_RE_SEG = re.compile(r"SEGMENT:\s*\[?(\d+)", re.I)
_RE_Q = re.compile(r"QUOTE:\s*(.+?)(?=\n\s*FEEDBACK:|\Z)", re.I | re.S)
_RE_FB = re.compile(r"FEEDBACK:\s*(.+)", re.I | re.S)

CRITIC_SYS = (
    "You are a meticulous math teacher reviewing a student's solution. "
    "The student's FINAL ANSWER IS WRONG. Find the FIRST place where the reasoning goes wrong.\n\n"
    "Rules:\n"
    "1. Identify exactly ONE segment -- the FIRST containing a genuine error.\n"
    "2. Quote the exact wrong sentence verbatim.\n"
    "3. Say briefly what is wrong and what to do instead.\n"
    "4. DO NOT continue the solution. DO NOT compute or state the final answer.\n\n"
    "Reply exactly as:\nSEGMENT: <index>\nQUOTE: <verbatim>\nFEEDBACK: <what is wrong and what to do>"
)
# ---- GT variant (CRITIC_OPD_GIVE_GT=1): the critic is given the correct answer ----
# The original prompt only tells the teacher "the student's answer is wrong" without giving the correct answer, so the teacher must first solve the problem itself to tell
# which step is wrong -- and Qwen3-4B itself scores only about 40 on aime24. Here the correct answer is written into the task definition,
# a method for locating the error is given, false positives such as "a detour the student later corrects" are ruled out, and the teacher may say "not found":
# a wrong location is worse than none, because it makes the student rewrite from a position that was actually correct.
_GT_RULES = (
    "How to locate the error:\n"
    "Read the numbered segments in order. For each one ask: given everything before it, is this "
    "step still correct, and does it still allow the problem to be solved? The FIRST segment "
    "where the answer is no is the error.\n\n"
    "Do NOT flag a segment merely because it is: an exploratory attempt the student later "
    "abandons or corrects; a longer route than necessary; or sloppy notation that does not change "
    "the mathematics. Flag only a step that is actually wrong.\n\n"
    "Rules you must follow:\n"
    "1. Identify exactly ONE segment -- the FIRST one containing a genuine error.\n"
    "2. Quote verbatim the single sentence in that segment that carries the error.\n"
    "3. Feedback: say what is wrong and what to do from that point instead. The student will "
    "resume writing immediately after this point, so phrase it as guidance they can act on. "
    "Two or three sentences at most.\n"
    "4. NEVER reveal the final answer. Do not state it, do not write an expression that "
    "evaluates to it, do not say what it should be. Describe the method, not the result.\n"
    "5. If every step looks correct to you and you cannot identify a genuine error, reply with "
    "SEGMENT: NONE. Do not guess -- a wrong location is worse than none.\n\n"
    "Reply in exactly this format:\n"
    "SEGMENT: <index, or NONE>\n"
    "QUOTE: <the exact wrong sentence, copied verbatim>\n"
    "FEEDBACK: <what is wrong and what to do instead>"
)
CRITIC_SYS_GT = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose "
    "FINAL ANSWER IS WRONG. You are given the correct answer. Use it as a reference to trace the "
    "student's work forward and find the EARLIEST step after which the correct answer can no "
    "longer be reached.\n\n" + _GT_RULES)
# An unfinished trajectory (hit the length cap) has no final answer, so we cannot say "the answer is wrong"; the teacher judges whether this path can still succeed.
CRITIC_SYS_GT_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an "
    "answer. You are given the correct answer. Use it as a reference to trace the student's work "
    "forward and find the EARLIEST step that sent the reasoning off track. If the work is still on "
    "a valid path and was merely slow, reply SEGMENT: NONE.\n\n" + _GT_RULES)
_RE_NONE = re.compile(r"SEGMENT:\s*\[?\s*(NONE|N/A|-)\b", re.I)

# ---- Reference-solution critique (CRITIC_OPD_REF_FILE): the critic also gets a complete correct solution produced offline by the teacher ----
# Offline check (criticopd/ref_probe.py, 400 wrong solutions from the same batch): repair success rate 22.0% -> 36.0%,
# 95% CI [+9.8, +18.1]; answer leakage under the strict criterion is unchanged (9.2% vs 9.2%), rewrites do not get shorter, and false positives drop.
# The prompt is verbatim identical to variant B in ref_probe.py.
_REF_RULES = (
    "How to locate the error:\n"
    "Read the student's numbered segments in order. Check the mathematics of each one, using the "
    "reference solution as a guide to what is true. The FIRST segment containing a genuinely wrong "
    "step -- a false statement, a wrong computation, an invalid deduction, or a wrong setup -- is "
    "the error.\n\n"
    "The student may solve the problem differently from the reference. A different approach is NOT "
    "an error: flag a segment only if its mathematics is actually wrong, never merely because it "
    "differs from the reference. Also do NOT flag an exploratory attempt the student later abandons "
    "or corrects, a longer route than necessary, or sloppy notation that does not change the "
    "mathematics.\n\n"
    "Rules you must follow:\n"
    "1. Identify exactly ONE segment -- the FIRST one containing a genuine error.\n"
    "2. Quote verbatim the single sentence in that segment that carries the error.\n"
    "3. Feedback: say concretely what is wrong at that step and what the correct reasoning for THAT "
    "step is (you may use the reference to state the correct method or fact for that step). The "
    "student will resume writing immediately after this point. Two or three sentences at most.\n"
    "4. NEVER reveal the final answer. Do not describe later steps of the reference and do not copy "
    "it.\n"
    "5. If every step looks correct to you, reply SEGMENT: NONE. Do not guess -- a wrong location is "
    "worse than none.\n\n")
_REF_FMT = ("Reply in exactly this format:\n"
            "SEGMENT: <index, or NONE>\n"
            "QUOTE: <the exact wrong sentence, copied verbatim>\n"
            "FEEDBACK: <what is wrong and what to do instead>")
CRITIC_SYS_REF = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose FINAL "
    "ANSWER IS WRONG. You are given the correct final answer and a correct reference solution. "
    "Use them to find the EARLIEST step where the student's work goes wrong.\n\n" + _REF_RULES + _REF_FMT)
# Unfinished trajectories: only 2.2% in the offline check, not tested separately; the opening reuses the wording of CRITIC_SYS_GT_CAPPED, the rest is the same as above.
CRITIC_SYS_REF_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an answer. "
    "You are given the correct final answer and a correct reference solution. Use them to find the "
    "EARLIEST step that sent the reasoning off track. If the work is still on a valid path and was "
    "merely slow, reply SEGMENT: NONE.\n\n" + _REF_RULES + _REF_FMT)

# ---- Truncation variant (CRITIC_OPD_TRUNC=1): the student stops right after fixing the wrong step ----
# The original design had the student write from the splice point all the way to EOS: offline, 67.8% of rewrites hit the length cap and the repair success rate was 0.0%, yet all of them
# went into the training sequence with mask=1. The erroneous segment itself has a median of only 149 characters (about 40 tokens), so we train only on "fix this step".
# Boundary = the first blank line after substantive content (consistent with the segmentation rule).
_BLANK = re.compile(r"\S[^\S\n]*\n[^\S\n]*\n")


def _cut_at_blank_line(tok, ids: list[int]) -> int:
    """Return k such that ids[:k] ends exactly at "the first blank line after substantive content" (including that blank line).
    "Whether the prefix already contains a blank line" is monotone in k, so use binary search; if there is no blank line, keep the whole span."""
    if not ids or not _BLANK.search(tok.decode(ids, skip_special_tokens=True)):
        return len(ids)
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if _BLANK.search(tok.decode(ids[:mid], skip_special_tokens=True)):
            hi = mid
        else:
            lo = mid + 1
    return lo


FEEDBACK_TMPL = ("\n\n[Teacher feedback] {fb}\n\nUsing this feedback, continue the solution from here. "
                 "Put your final answer in \\boxed{{}}.\n\n")
# Refinement instruction of the Critique-GRPO baseline, identical to the official implementation (critique-GRPO/verl/critique_grpo/refinement_prompts.py):
# start a new conversation whose user message gives the problem, the original solution, and the critique, and asks for a step-by-step re-answer.
# (In a smoke test, appending the critique to the original answer and continuing in the same assistant turn made the student add only a one-line final answer, so refinement degenerated into copying the answer)
CGRPO_REFINE_SYS = "You are a helpful assistant."
CGRPO_REFINE_USER = (
    "Given the following inputs:\n"
    "**Question**: {question}\n"
    "**Previous Solution**: {failed}\n"
    "**Critique**: {critique}\n\n"
    "Please re-answer by:\n"
    "- Correcting potential errors identified in the critique, if they exist.\n"
    "- Providing clear, step-by-step reasoning.\n"
    "- Formatting the final answer as `\\boxed{{answer}}`.\n\n"
    "Ensure the revised solution addresses all issues raised in the critique.")


def _is_leadin(p):
    return bool(_LEADIN.match(p)) or (len(p) <= 12 and p.rstrip().endswith(":"))


def segment(text):
    """Split on blank lines, then merge headings/separators/lead-in sentences into the following segment. Decide by form, not length --
    merging only by "short" would also swallow normal short content blocks (a one-line formula, a one-sentence conclusion)."""
    parts = [p.strip() for p in _SEP.split(text) if p.strip()]
    merged, buf = [], []
    for p in parts:
        if _is_leadin(p):
            buf.append(p); continue
        merged.append("\n\n".join(buf + [p]) if buf else p); buf = []
    if buf: merged.append("\n\n".join(buf))
    return merged or parts


def parse_critic(text, segs):
    """Locate by the quote in the original text first and use the index only as a fallback -- the critic's index is often off by one or two, the quote is not."""
    if _RE_NONE.search(text):          # critic explicitly says it found no error: give up, do not let quote matching override this
        return None, None
    mq, mf, ms = _RE_Q.search(text), _RE_FB.search(text), _RE_SEG.search(text)
    fb = mf.group(1).strip() if mf else None
    quote = mq.group(1).strip().strip('"') if mq else None
    idx = None
    if quote and len(quote) >= 8:
        norm = lambda t: re.sub(r"\s+", " ", t).strip()
        nq = norm(quote)[:80]
        for i, s in enumerate(segs):
            if quote[:60] in s or nq in norm(s):
                idx = i; break
    if idx is None and ms:
        j = int(ms.group(1))
        if 0 <= j < len(segs): idx = j
    return idx, fb



# ---- k ablation (CRITIC_OPD_KERR): the critic teacher lists all errors in order, and we cut at the k-th error ----
# Same rules as CRITIC_SYS_GT, except "find only the first" becomes "list all of them in order (at most 5)".
# With k=1 the feedback format is exactly that of R4GT (feedback on the first error only), so k=1 differs from R4GT only in the critique prompt itself.
_MULTI_RULES = (
    "How to locate the errors:\n"
    "Read the numbered segments in order. For each one ask: given everything before it, is this step "
    "correct? List every segment that contains a genuine error, in the order in which they appear.\n\n"
    "Do NOT flag a segment merely because it is: an exploratory attempt the student later abandons or "
    "corrects; a longer route than necessary; or sloppy notation that does not change the mathematics. "
    "Flag only a step that is actually wrong.\n\n"
    "Rules you must follow:\n"
    "1. List the erroneous segments in order of appearance, at most 5.\n"
    "2. For each one, quote verbatim the single sentence that carries the error.\n"
    "3. For each one, say what is wrong and what to do instead. The student will resume writing after "
    "the last listed error, so phrase it as guidance they can act on. Two or three sentences per error.\n"
    "4. NEVER reveal the final answer. Do not state it, do not write an expression that evaluates to it, "
    "do not say what it should be. Describe the method, not the result.\n"
    "5. If every step looks correct to you and you cannot identify a genuine error, reply with "
    "SEGMENT: NONE. Do not guess -- a wrong location is worse than none.\n\n"
    "Reply in exactly this format, one block per error:\n"
    "ERROR 1\nSEGMENT: <index>\nQUOTE: <the exact wrong sentence, copied verbatim>\n"
    "FEEDBACK: <what is wrong and what to do instead>\n"
    "ERROR 2\nSEGMENT: <index>\nQUOTE: <...>\nFEEDBACK: <...>")
CRITIC_SYS_MULTI = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose FINAL ANSWER "
    "IS WRONG. You are given the correct answer. Use it as a reference to trace the student's work "
    "forward and find the steps that contain genuine errors.\n\n" + _MULTI_RULES)
CRITIC_SYS_MULTI_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an answer. "
    "You are given the correct answer. Use it as a reference to trace the student's work forward and find "
    "the steps that sent the reasoning off track. If the work is still on a valid path and was merely "
    "slow, reply SEGMENT: NONE.\n\n" + _MULTI_RULES)
# Ablation CRITIC_OPD_FB_MODE=generic: same template as the method, but with a single content-free feedback message
GENERIC_FEEDBACK = "Your solution contains errors. Check the previous reasoning and redo it."
# For ablation CRITIC_OPD_ALL: for a correct rollout we cannot say "the answer is wrong", so the first paragraph is neutral; the other rules are unchanged
CRITIC_SYS_MULTI_ANY = (
    "You are a meticulous math teacher. A student has submitted a complete solution. You are given the correct "
    "answer. Use it as a reference to trace the student's work forward and find the steps that contain genuine "
    "errors.\n\n" + _MULTI_RULES)
_RE_ERRHDR = re.compile(r"\bERROR\s*\d+\s*:?\s*$", re.I)


def parse_critic_multi(text, segs, max_err=5):
    """Parse a multi-error critique -> [(segment index, quote, feedback)], sorted by segment index and deduplicated.
    Split into blocks at "SEGMENT:" (this also works when the critic writes the error number and segment index on the same line); each block reuses parse_critic's
    quote-first localization; a leftover "ERROR k" header of the next block at the end of a block is removed from the feedback. Returns an empty list when the whole text is only NONE."""
    errs, seen = [], set()
    for b in re.split(r"(?i)(?=SEGMENT\s*:)", text):
        if not re.match(r"(?i)SEGMENT\s*:", b):
            continue
        idx, fb = parse_critic(b, segs)
        if idx is None or not fb or idx in seen:
            continue
        fb = "\n".join(ln for ln in fb.strip().splitlines() if not _RE_ERRHDR.search(ln)).strip()
        if not fb:
            continue
        mq = _RE_Q.search(b)
        q = mq.group(1).strip().strip('"') if mq else ""
        errs.append((idx, q, fb)); seen.add(idx)
    errs.sort(key=lambda e: e[0])
    return errs[:max_err]


_RE_LAST_BLOCK = re.compile(r"(?i)(?:ERROR\s*\d+\s*:?\s*)?SEGMENT\s*:")


def drop_incomplete_tail(text):
    """When the critique output hits the length cap, the last block (from the last SEGMENT: on) is necessarily unfinished: drop the whole block.
    Call only when truncation is confirmed; if there is no SEGMENT: at all, return the text unchanged."""
    ms = list(_RE_LAST_BLOCK.finditer(text))
    return text[:ms[-1].start()] if ms else text


def kerr_select(errs, kerr):
    """Cut at the k-th error; if there are fewer than k errors, take the last one. kerr is '1'/'2'/'3'/.../'last'."""
    if not errs:
        return []
    k = len(errs) if str(kerr) == "last" else min(int(kerr), len(errs))
    return errs[:k]


def leaks_answer(fb, gt, stu):
    """The reference answer appears in the feedback (as a standalone token) but not in the student's original text -> treat as answer leakage."""
    if gt is None:
        return False
    g = str(gt).strip().strip("$")
    if not g:
        return False
    # Must not be directly adjacent to an alphanumeric or to a decimal point + digit (avoids false matches in 3280, 1.328, 23.5); a sentence-final "23." still counts
    pat = r"(?<![\w.])" + re.escape(g) + r"(?!\w|\.\d)"
    return bool(re.search(pat, fb or "")) and not re.search(pat, stu or "")


def drop_leaky(errs, gt, stu):
    """Drop errors whose feedback leaks the answer. Later entries of a multi-error critique are more likely to state the answer in the feedback (k=1/2/last about 1.5%/7%/10%);
    without dropping them, a larger k makes the rewrite more likely to "copy the answer", and the comparison across k is unfair."""
    return [e for e in errs if not leaks_answer(e[2], gt, stu)]


def kerr_feedback(sel):
    """With k=1 the feedback format is exactly that of R4GT; with k>1 the errors are listed in order (each with its quote from the original text, truncated to 200 characters)."""
    if len(sel) == 1:
        return sel[0][2]
    parts = [f'({i + 1}) In "{q[:200]}": {fb}' for i, (_, q, fb) in enumerate(sel)]
    return f"Your solution contains {len(sel)} errors.\n" + "\n".join(parts)


def _tok_prefix_for_text(tok, ids: list[int], text: str) -> int:
    """Return the smallest k such that decode(ids[:k]) already covers text.

    Previously the kept segments were decoded to text and re-encoded to tokens, which introduces tokenization drift:
    the training sequence is no longer a prefix of the student's own trajectory, and out.log_probs no longer lines up with kept_ids --
    in bypass mode rollout_log_probs directly become old_log_probs (the denominator of the PPO ratio),
    so the misalignment corrupts training. Binary-searching the cut directly on the original rollout tokens removes both problems.
    """
    target = len(text.rstrip())
    if target <= 0:
        return 0
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=True).rstrip()) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---- In-process grading -> grading in separate processes ----
# An R4GTV run once froze entirely for 47 minutes at step 2: the grading thread was stuck in sympy's
# Integer._eval_power (C-level big-integer exponentiation that holds the GIL for the whole computation), so the agent loop's event-loop thread
# could not get the GIL and even the 30-second asyncio.wait_for timeout could not fire. Thread-based grading cannot handle this case.
# Here grading runs in persistent subprocesses, one request and one reply per line over stdin/stdout; the parent only reads the pipe asynchronously,
# and on timeout kills the subprocess and spawns a replacement. A hung subprocess only blocks itself.
_GRADER_SRC = r"""
import sys, os, json
out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                         # library prints all go to stderr; the protocol uses only out
sys.path.insert(0, "__ROOT__")
os.environ.setdefault("MATH_GRADER_PATH", "__ROOT__/opd/reward/grader")
from opd.reward.math_reward import compute_score
out.write("@@READY\n")
for line in sys.stdin:
    try:
        q = json.loads(line)
        v = compute_score(solution_str=q["t"], ground_truth=q["g"])
        r = float(v["score"] if isinstance(v, dict) else v) > 0.5
    except Exception:
        r = False
    out.write("@@R " + ("1" if r else "0") + "\n")
""".replace("__ROOT__", _ROOT)


class _GraderPool:
    _pools: dict = {}

    def __init__(self, n: int):
        self.n = n
        self.idle: asyncio.Queue = asyncio.Queue()
        self.started = False
        self.lock = asyncio.Lock()

    @classmethod
    def get(cls) -> "_GraderPool":
        loop = asyncio.get_running_loop()
        p = cls._pools.get(id(loop))
        if p is None:
            p = cls._pools[id(loop)] = cls(int(os.environ.get("CRITIC_OPD_GRADER_PROCS", "8")))
        return p

    async def _spawn(self):
        import sys as _sys
        proc = await asyncio.create_subprocess_exec(
            _sys.executable, "-c", _GRADER_SRC, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=180)
            if not line:
                raise RuntimeError("grader subprocess failed to start")
            if line.startswith(b"@@READY"):
                return proc

    async def _replace(self, proc):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        asyncio.ensure_future(proc.wait())            # reap it to avoid a zombie process
        for _ in range(5):
            try:
                self.idle.put_nowait(await self._spawn())
                return
            except Exception as e:                    # noqa: BLE001
                logger.warning("[critic_opd] failed to respawn grader subprocess: %s", e)
                await asyncio.sleep(5)

    async def _ensure(self):
        if self.started:
            return
        async with self.lock:
            if self.started:
                return
            for proc in await asyncio.gather(*[self._spawn() for _ in range(self.n)]):
                self.idle.put_nowait(proc)
            self.started = True

    async def grade(self, text: str, gt, timeout: float) -> bool:
        await self._ensure()
        proc = await self.idle.get()
        try:
            proc.stdin.write((json.dumps({"t": text, "g": str(gt)}) + "\n").encode())
            await proc.stdin.drain()

            async def _read():
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        raise RuntimeError("grader subprocess exited")
                    if line.startswith(b"@@R "):
                        return line[4:5] == b"1"

            r = await asyncio.wait_for(_read(), timeout=timeout)
        except BaseException:
            # Timeout / cancellation / dead subprocess: this subprocess may still emit a late result line,
            # which the next request would read and misattribute, so always kill it and spawn a new one.
            asyncio.ensure_future(self._replace(proc))
            raise
        self.idle.put_nowait(proc)
        return r


@register("critic_opd_agent")
class CriticOpdAgentLoop(AgentLoopBase):
    # Class-level defaults for the new switches, so they are defined even on construction paths that bypass __init__. Otherwise reading a missing
    # attribute in _run raises AttributeError, which the outer try/except in run() silently downgrades to a plain rollout --
    # the same kind of silent failure as the earlier missing asyncio import.
    give_gt = False
    trunc = False
    trunc_max = 512
    keep_correct = False
    ref_file = None
    critic_maxlen = 34817
    critic_tokens_exact = False
    rewriter = "student"
    drop_wrong = False
    fb_answer = False
    kerr = None
    prefix_loss = False
    critic_all = False
    cut_mode = "error"
    fb_mode = "critique"
    _REFS = None            # {sha1(problem text): reference solution}, loaded once per process
    anneal = None
    global_step = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # SPLICE=before is "option 1" (drop the erroneous segment), =after is "option 2" (keep the erroneous segment so the student can reflect on it)
        self.splice = os.environ.get("CRITIC_OPD_SPLICE", "before").strip().lower()
        self.enabled = os.environ.get("CRITIC_OPD_ENABLE", "1") == "1"
        self.use_feedback = os.environ.get("CRITIC_OPD_USE_FEEDBACK", "1") == "1"  # 0 = the no-feedback control of R2
        self.critic_max_tokens = int(os.environ.get("CRITIC_OPD_CRITIC_TOKENS", "512"))
        self.give_gt = os.environ.get("CRITIC_OPD_GIVE_GT", "0") == "1"
        self.trunc = os.environ.get("CRITIC_OPD_TRUNC", "0") == "1"
        self.trunc_max = int(os.environ.get("CRITIC_OPD_TRUNC_MAX", "512"))
        # Keep only correct rewrites: grade the rewrite against gt again; if it is not fixed, fall back to the original rollout and train it with plain OPD.
        self.keep_correct = os.environ.get("CRITIC_OPD_KEEP_CORRECT", "0") == "1"
        self.ref_file = os.environ.get("CRITIC_OPD_REF_FILE") or None
        self.critic_maxlen = int(os.environ.get("CRITIC_OPD_CRITIC_MAXLEN", "34817"))   # the teacher's max_model_len
        # Tuning (critique output cap): multi-error mode raises the cap to at least 1024 by default; set to 1 to use exactly CRITIC_OPD_CRITIC_TOKENS (may be below 1024)
        self.critic_tokens_exact = os.environ.get("CRITIC_OPD_CRITIC_TOKENS_EXACT", "0") == "1"
        # Ablation: the teacher writes the rewrite (same context, budget, and sampling temperature as the student rewrite); everything else (no loss on the prefix, the teacher scores with the critique) is unchanged
        self.rewriter = os.environ.get("CRITIC_OPD_REWRITER", "student").strip().lower()
        # Baseline: grade against the reference answer; wrong or length-capped rollouts get no loss and no critique (OPD on correct rollouts only)
        self.drop_wrong = os.environ.get("CRITIC_OPD_DROP_WRONG", "0") == "1"
        # Ablation: the critique leaks the answer (errors that leak the answer are not filtered, and the reference answer is stated at the end of the feedback)
        self.fb_answer = os.environ.get("CRITIC_OPD_FB_ANSWER", "0") == "1"
        # k ablation: '1'/'2'/'3'/'last'; when unset, behavior is exactly as before. Only meaningful when the erroneous segment is kept.
        self.kerr = (os.environ.get("CRITIC_OPD_KERR", "").strip().lower() or None)
        # Loss on the prefix: student tokens before the segment with the first error are also trained with plain OPD (the teacher does not see the feedback); when unset, they are all masked as before
        self.prefix_loss = os.environ.get("CRITIC_OPD_PREFIX_LOSS", "0") == "1"
        # Ablation: remove the grading gate so correct rollouts also go to the critic (with a neutral prompt; with no genuine error it replies NONE and the rollout is trained with plain OPD)
        self.critic_all = os.environ.get("CRITIC_OPD_ALL", "0") == "1"
        # Mechanism ablations: CUT=end keeps the whole wrong solution (no cut at the error location; length-capped ones are still cut at the error, otherwise there is no room to continue);
        # FB_MODE=generic keeps the cut point but replaces the feedback with a content-free generic message (which is also what the teacher sees when scoring)
        self.cut_mode = os.environ.get("CRITIC_OPD_CUT", "error").strip().lower()
        self.fb_mode = os.environ.get("CRITIC_OPD_FB_MODE", "critique").strip().lower()
        # Annealing (DAgger-style beta schedule): "decay start step,decay end step,beta floor". A wrong sample goes through critic repair with probability beta;
        # otherwise its own rollout is kept and trained with plain OPD. When unset, beta is always 1 (i.e., R4GT).
        _a = os.environ.get("CRITIC_OPD_ANNEAL", "").strip()
        self.anneal = tuple(float(x) for x in _a.replace(":", ",").split(",")) if _a else None   # use ":" as the separator inside sbatch --export
        self.global_step = int(kwargs.get("critic_global_step", -1) or -1)
        self._grader = None
        # Baseline OPSD-T: the student sees only the problem; the teacher sees an extra line with the reference answer after the problem and scores the whole rollout (no critique, no rewrite)
        self.opsd = os.environ.get("CRITIC_OPD_OPSD", "0") == "1"
        # Baseline Critique-GRPO: n_init initial responses plus 1 refinement slot per problem (ROLLOUT_N = n_init + 1).
        # Critique and refinement happen only when all initial responses in a group are wrong; otherwise the refinement slot is an empty row (all-zero mask, excluded from the advantage baseline)
        self.cgrpo = os.environ.get("CRITIC_OPD_CGRPO", "0") == "1"
        self.cgrpo_n_init = int(os.environ.get("CRITIC_OPD_CGRPO_N_INIT", "7"))
        self.cgrpo_wait = float(os.environ.get("CRITIC_OPD_CGRPO_WAIT", "5400"))
        self.rollout_max_len = int(getattr(self.rollout_config, "max_model_len", None) or 34817)

    @staticmethod
    def _disarm_sigalrm():
        """The grader can use SIGALRM safely only in the main thread; the agent loop's event loop runs in a non-main thread.

        openmathinst_utils.time_limit does, in order,
            signal.setitimer(ITIMER_REAL, s)   # process-wide, succeeds in any thread
            signal.signal(SIGALRM, handler)    # raises ValueError in a non-main thread
        The exception happens before the yield, so the "disarm the timer" in finally never runs -> 10 seconds later SIGALRM
        is delivered with its default action (terminate the process). No traceback, no core dump; the raylet only sees
        "Worker unexpectedly exits with a connection error code 2".
        The math_verify source code documents the same issue and recommends turning off its timeout.

        Here the whole SIGALRM chain becomes a no-op: the timer is never actually armed (so no signal is
        ever delivered), and signal.signal(SIGALRM, ...) no longer raises (so finally completes normally and
        grading behaves exactly as offline). Other signals are untouched, so Ray's own SIGINT/SIGTERM handling is unaffected.
        """
        import signal as _sg
        if getattr(_sg, "_critic_opd_disarmed", False):
            return
        _orig = _sg.signal

        def _sig(sig, handler):
            if sig == _sg.SIGALRM:
                return _sg.SIG_DFL
            return _orig(sig, handler)

        _sg.signal = _sig
        _sg.setitimer = lambda *a, **k: (0.0, 0.0)
        _sg.alarm = lambda *a, **k: 0
        _sg._critic_opd_disarmed = True

    def _grade(self, text, gt):
        if self._grader is None:
            import sys
            sys.path.insert(0, _ROOT)
            os.environ.setdefault("MATH_GRADER_PATH", os.path.join(_ROOT, "opd", "reward", "grader"))
            self._disarm_sigalrm()
            from opd.reward.math_reward import compute_score
            self._grader = compute_score
        try:
            v = self._grader(solution_str=text, ground_truth=gt)
            return float(v["score"] if isinstance(v, dict) else v) > 0.5
        except Exception:
            return False

    def _reference(self, question: str):
        if not self.ref_file:
            return None
        if type(self)._REFS is None:
            import hashlib as _hl, json as _js
            type(self)._REFS = _js.load(open(self.ref_file))
            type(self)._sha1 = staticmethod(lambda t: _hl.sha1(t.encode()).hexdigest())
            logger.warning("[critic_opd] loaded %d reference solutions: %s", len(type(self)._REFS), self.ref_file)
        return type(self)._REFS.get(type(self)._sha1(question))

    def _beta(self) -> float:
        if not self.anneal or self.global_step < 0:
            return 1.0
        t0, t1, bmin = self.anneal
        if self.global_step <= t0:
            return 1.0
        if self.global_step >= t1:
            return bmin
        return 1.0 - (1.0 - bmin) * (self.global_step - t0) / max(t1 - t0, 1e-9)

    async def _verify_async(self, text, gt) -> bool:
        """Check whether the rewrite is fixed. Timeouts / errors count as "not fixed" -- the conservative direction is to keep questionable rewrites out of training."""
        try:
            return bool(await asyncio.wait_for(_GraderPool.get().grade(text, gt, 30), timeout=240))
        except (TimeoutError, asyncio.TimeoutError, RuntimeError, OSError) as e:
            logger.warning("[critic_opd] rewrite verification failed (%s), treating it as not fixed", type(e).__name__)
            return False

    async def _grade_async(self, text, gt):
        """Decide whether to trigger the critic. Timeouts / errors count as "correct" (critic not triggered), which is the conservative direction."""
        try:
            return bool(await asyncio.wait_for(_GraderPool.get().grade(text, gt, 30), timeout=240))
        except (TimeoutError, asyncio.TimeoutError, RuntimeError, OSError) as e:
            logger.warning("[critic_opd] grading failed (%s), treating it as correct (critic not triggered)", type(e).__name__)
            return True
    async def _teacher_generate(self, prompt_ids, max_tokens, temperature=0.0, top_p=1.0):
        """Let the teacher generate freely on the given prompt. The teacher client was originally used only for scoring
        (max_tokens=1 + prompt_logprobs); with a different set of sampling_params it generates freely. Critiques use greedy decoding;
        teacher rewrites (ablation) use the same temperature as the student rollout."""
        tm = self.teacher_manager
        if tm is None: return None
        key = next(iter(tm.teacher_model_configs))
        out = await tm.teacher_client[key].generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params={"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p},
        )
        return out.token_ids

    _CNT = {"n": 0, "capped": 0, "correct": 0, "attempted": 0, "parsed": 0,
            "repaired": 0, "exc": 0}

    def _dump(self, rec):
        """For analysis (mismatch and reliability in Figure 2): written only if CRITIC_OPD_DUMP_DIR is set, one jsonl per process."""
        d = os.environ.get("CRITIC_OPD_DUMP_DIR")
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/agent_{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _tally(self, **kw):
        c = type(self)._CNT
        c["n"] += 1
        for k in kw:
            c[k] = c.get(k, 0) + 1
        if c["n"] % 64 == 0:
            logger.warning("[critic_opd] samples=%d capped=%d correct=%d attempted=%d parsed=%d repaired=%d exceptions=%d",
                           c["n"], c["capped"], c["correct"], c["attempted"],
                           c["parsed"], c["repaired"], c["exc"])
            print(f"[critic_opd] tally {c}", flush=True)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        try:
            return await self._run(sampling_params, **kwargs)
        except Exception:
            # When a Ray actor dies abruptly the exception does not propagate; the driver only sees ActorDiedError. Print it first, then decide.
            import traceback
            self._tally(exc=1)
            logger.error("[critic_opd] run() raised an exception, falling back to a plain rollout:\n%s", traceback.format_exc())
            print("[critic_opd] EXCEPTION\n" + traceback.format_exc(), flush=True)
            if self.cgrpo and int(kwargs.get("cgrpo_slot", -1)) >= self.cgrpo_n_init:
                return await self._cgrpo_empty(list(kwargs["raw_prompt"]), {"cgrpo_exc": 1})
            return await self._plain(sampling_params, **kwargs)

    async def _run_opsd(self, sampling_params, messages, mm, mmk, prompt_ids, **kwargs) -> AgentLoopOutput:
        """OPSD-T (Penaloza et al. 2026; Zhao et al. 2026, with the teacher replaced by an external teacher):
        the student samples on a prompt with only the problem; when scoring, the teacher's user message ends with an extra line giving the reference answer, and the whole rollout is scored under it.
        Reuses the L_fb scoring path: critic_fb_prompt_ids = the prompt with the answer, n_kept = 0, n_repair = the whole response."""
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            out: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids,
                sampling_params=sampling_params, mm_processor_kwargs=mmk)
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        resp = out.token_ids[:rlen]
        gt = None
        rm = kwargs.get("reward_model")
        if isinstance(rm, dict): gt = rm.get("ground_truth")
        stats, fb = {"opsd": 1}, {}
        if gt is not None and len(resp) > 0:
            tmsg = [dict(m) for m in messages]
            tmsg[-1]["content"] = f"{tmsg[-1]['content']}\n\n[Reference answer: {gt}]"
            tids = await self.apply_chat_template(
                tmsg, images=mm.get("images"), videos=mm.get("videos"),
                audios=mm.get("audios"), mm_processor_kwargs=mmk)
            fb = {"critic_fb_prompt_ids": list(tids), "critic_fb_n_kept": 0, "critic_fb_n_repair": len(resp)}
            if not type(self)._DBG.get("opsd"):
                type(self)._DBG["opsd"] = 1
                print("[opsd] student prompt tail:", repr(self.tokenizer.decode(prompt_ids)[-200:]), flush=True)
                print("[opsd] teacher prompt tail:", repr(self.tokenizer.decode(tids)[-260:]), flush=True)
            # Metrics only: fraction correct; answering directly with little reasoning (correct and at most 64 tokens); the response mentions the reference answer
            text = self.tokenizer.decode(resp, skip_special_tokens=True)
            capped = len(out.token_ids) >= rlen
            ok = (not capped) and await self._grade_async(text, gt)
            short = bool(ok and len(resp) <= 64)
            ref = bool(re.search(r"reference answer|\[reference", text, re.I))
            stats.update({"opsd_correct": int(ok), "opsd_short_answer": int(short), "opsd_mentions_ref": int(ref)})
            self._tally(**{k: 1 for k, v in (("opsd_correct", ok), ("opsd_short_answer", short),
                                             ("opsd_mentions_ref", ref), ("capped", capped)) if v})
        else:
            self._tally(opsd_no_gt=1)
        metrics.setdefault("num_preempted", out.num_preempted if out.num_preempted is not None else -1)
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
            response_logprobs=out.log_probs[:rlen] if out.log_probs is not None else None,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2, metrics=metrics,
            extra_fields={**(out.extra_fields or {}), "critic_opd": stats, **fb})

    # Within-group coordination for Critique-GRPO: the n_init + 1 copies of a problem share a uid and are in the same AgentLoopWorker
    # (repeat is interleaved and chunks are contiguous, so each chunk holds a whole number of groups). Initial responses register their results here,
    # and the refinement slot waits until the whole group has registered before deciding whether to refine.
    _CG = {}
    _DBG = {}

    def _cg_entry(self, uid):
        e = type(self)._CG.get(uid)
        if e is None:
            e = {"res": {}, "ev": asyncio.Event()}
            type(self)._CG[uid] = e
        return e

    async def _cgrpo_empty(self, messages, stats, prompt_ids=None, mm=None, mmk=None) -> AgentLoopOutput:
        """Empty row when not refining: 1 token with an all-zero mask; it is excluded from the group mean in advantage estimation and does not appear in the loss."""
        if prompt_ids is None:
            mm = await self.process_multi_modal_info(messages)
            mmk = self._get_mm_processor_kwargs(mm.get("audios"))
            prompt_ids = await self.apply_chat_template(
                messages, images=mm.get("images"), videos=mm.get("videos"),
                audios=mm.get("audios"), mm_processor_kwargs=mmk)
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=[self.tokenizer.eos_token_id], response_mask=[0],
            response_logprobs=[0.0], multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2,
            metrics={"num_preempted": -1}, extra_fields={"critic_opd": stats, "cgrpo_role": 2})

    async def _critique_text(self, messages, text, gt, capped):
        """The same critique as the final CriticOPD version: a multi-error critique with the reference answer (at most 5 entries), dropping the unfinished last entry;
        with CRITIC_OPD_FB_ANSWER=1, entries that leak the answer are not filtered and the answer is given at the end (same as critans). Returns None when no usable entry remains."""
        segs = segment(text)
        body = "\n\n".join(f"[{i}] {seg}" for i, seg in enumerate(segs))
        cmsg = [{"role": "system", "content": CRITIC_SYS_MULTI_CAPPED if capped else CRITIC_SYS_MULTI},
                {"role": "user", "content":
                 f"Problem:\n{messages[-1]['content']}\n\n"
                 f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                 f"Student's solution, split into numbered segments:\n\n{body}"}]
        cids = self.tokenizer.encode(self.tokenizer.apply_chat_template(
            cmsg, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
        cmax = self.critic_max_tokens if self.critic_tokens_exact else max(1024, self.critic_max_tokens)
        if len(cids) + cmax > self.critic_maxlen:
            return None
        ctok = await self._teacher_generate(cids, cmax)
        if not ctok:
            return None
        ctext = self.tokenizer.decode(ctok, skip_special_tokens=True)
        if len(ctok) >= cmax:
            ctext = drop_incomplete_tail(ctext)
        errs = parse_critic_multi(ctext, segs)
        if not self.fb_answer:
            errs = drop_leaky(errs, gt, text)
        sel = kerr_select(errs, self.kerr or "last")
        if not sel:
            return None
        fb = kerr_feedback(sel)
        if self.fb_answer:
            fb = f"{fb}\nThe correct final answer is {gt}."
        return fb

    async def _run_cgrpo(self, sampling_params, messages, mm, mmk, prompt_ids, **kwargs) -> AgentLoopOutput:
        """Sampling part of Critique-GRPO (Zhang et al. 2025). Slots 0..n_init-1 are ordinary initial responses;
        the last slot waits until all initial responses in the group are done: if all are wrong, it critiques each one, re-answers in a new conversation with the official instruction, and takes a correct refinement
        (or a random one if none is correct) as this row's response; otherwise it returns an empty row. The training sequence contains only the problem and the refined response, not the critique."""
        uid = str(kwargs.get("uid"))
        slot = int(kwargs.get("cgrpo_slot", -1))
        n0 = self.cgrpo_n_init
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        gt = None
        rm = kwargs.get("reward_model")
        if isinstance(rm, dict): gt = rm.get("ground_truth")

        if slot < n0:
            ent = self._cg_entry(uid)
            rec = None
            try:
                metrics = {}
                with simple_timer("generate_sequences", metrics):
                    out: TokenOutput = await self.server_manager.generate(
                        request_id=uuid4().hex, prompt_ids=prompt_ids,
                        sampling_params=sampling_params, mm_processor_kwargs=mmk)
                resp = out.token_ids[:rlen]
                capped = len(out.token_ids) >= rlen
                text = self.tokenizer.decode(resp, skip_special_tokens=True)
                ok = (not capped) and gt is not None and await self._grade_async(text, gt)
                rec = (text, capped, bool(ok))
            finally:
                ent["res"][slot] = rec            # register even on error, otherwise the refinement slot keeps waiting
                if len(ent["res"]) >= n0:
                    ent["ev"].set()
            self._tally(cgrpo_init=1, **({"correct": 1} if ok else {}), **({"capped": 1} if capped else {}))
            metrics.setdefault("num_preempted", out.num_preempted if out.num_preempted is not None else -1)
            return AgentLoopOutput(
                prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
                response_logprobs=out.log_probs[:rlen] if out.log_probs is not None else None,
                multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2, metrics=metrics,
                extra_fields={**(out.extra_fields or {}), "critic_opd": {"cgrpo_slot": slot}, "cgrpo_role": 0})

        ent = self._cg_entry(uid)
        stats = {"cgrpo_slot": slot}
        try:
            await asyncio.wait_for(ent["ev"].wait(), timeout=self.cgrpo_wait)
        except (TimeoutError, asyncio.TimeoutError):
            stats["cgrpo_timeout"] = 1
            self._tally(cgrpo_timeout=1)
        res = dict(ent["res"])
        type(self)._CG.pop(uid, None)
        recs = [res.get(i) for i in range(n0)]
        all_wrong = len(res) >= n0 and all(r is not None and not r[2] for r in recs)
        if not all_wrong or gt is None or self.teacher_manager is None:
            self._tally(cgrpo_skip=1)
            return await self._cgrpo_empty(messages, stats, prompt_ids, mm, mmk)
        self._tally(cgrpo_trigger=1)

        async def refine(rec):
            text, capped, _ = rec
            fb = await self._critique_text(messages, text, gt, capped)
            if not fb:
                return None
            rmsg = [{"role": "system", "content": CGRPO_REFINE_SYS},
                    {"role": "user", "content": CGRPO_REFINE_USER.format(
                        question=messages[-1]["content"], failed=text, critique=fb)}]
            rids = await self.apply_chat_template(
                rmsg, images=mm.get("images"), videos=mm.get("videos"),
                audios=mm.get("audios"), mm_processor_kwargs=mmk)
            if not type(self)._DBG.get("cgrpo"):
                type(self)._DBG["cgrpo"] = 1
                print("[cgrpo] refine prompt tail:", repr(self.tokenizer.decode(rids)[-900:]), flush=True)
            budget = min(rlen, self.rollout_max_len - len(rids) - 1)
            if budget < 256:
                return None
            r: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=list(rids),
                sampling_params={**sampling_params, "max_tokens": budget}, mm_processor_kwargs=mmk)
            ids = list(r.token_ids)[:rlen]
            rcapped = len(r.token_ids) >= budget
            ok = (not rcapped) and await self._verify_async(
                self.tokenizer.decode(ids, skip_special_tokens=True), gt)
            lp = list(r.log_probs[:len(ids)]) if r.log_probs is not None else [0.0] * len(ids)
            if len(lp) < len(ids):
                lp += [0.0] * (len(ids) - len(lp))
            return ids, lp, bool(ok)

        outs = await asyncio.gather(*[refine(r) for r in recs], return_exceptions=True)
        cands = [o for o in outs if isinstance(o, tuple) and len(o[0]) > 0]
        stats["cgrpo_n_refined"] = len(cands)
        stats["cgrpo_n_refined_ok"] = sum(o[2] for o in cands)
        if not cands:
            self._tally(cgrpo_no_refine=1)
            return await self._cgrpo_empty(messages, stats, prompt_ids, mm, mmk)
        good = [o for o in cands if o[2]]
        ids, lp, ok = random.choice(good or cands)
        stats["cgrpo_refine_ok"] = int(ok)
        stats["cgrpo_refine_len"] = len(ids)
        _n = type(self)._DBG.get("cgrpo_pick", 0)
        if _n < 3:
            type(self)._DBG["cgrpo_pick"] = _n + 1
            print(f"[cgrpo] picked refinement ({len(ids)} tok, ok={ok}; candidate lens {[len(o[0]) for o in cands]}):",
                  repr(self.tokenizer.decode(ids)[:600]), flush=True)
        self._tally(cgrpo_refined=1, **({"cgrpo_refine_ok": 1} if ok else {}))
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=ids, response_mask=[1] * len(ids), response_logprobs=lp,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2, metrics={"num_preempted": -1},
            extra_fields={"critic_opd": stats, "cgrpo_role": 1})

    async def _plain(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Safe fallback on errors: run once the single_turn way, without critic/repair."""
        messages = list(kwargs["raw_prompt"])
        mm = await self.process_multi_modal_info(messages)
        mmk = self._get_mm_processor_kwargs(mm.get("audios"))
        prompt_ids = await self.apply_chat_template(
            messages, images=mm.get("images"), videos=mm.get("videos"),
            audios=mm.get("audios"), mm_processor_kwargs=mmk)
        out: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params=sampling_params, mm_processor_kwargs=mmk)
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        resp = out.token_ids[:rlen]
        # The fallback path must also carry response_logprobs: _postprocess does, over the whole batch,
        #   torch.cat([input.response_logprobs for input in inputs])
        # and if even one sample in the batch is None it raises
        #   "TypeError: expected Tensor as element N in argument 0, but got NoneType".
        # An R3v2 run hit exactly this at step 80, and the whole job ended with rc=1
        # (the checkpoint had already been written, so no data was lost, but the job failed).
        _lp = list(out.log_probs[:len(resp)]) if out.log_probs is not None else None
        if _lp is not None and len(_lp) < len(resp):
            _lp += [0.0] * (len(resp) - len(_lp))
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
            response_logprobs=_lp,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2,
            metrics={"num_preempted": -1},
            extra_fields={**(out.extra_fields or {}), "critic_opd": {"fallback": 1}})

    async def _run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        mm = await self.process_multi_modal_info(messages)
        mmk = self._get_mm_processor_kwargs(mm.get("audios"))
        prompt_ids = await self.apply_chat_template(
            messages, images=mm.get("images"), videos=mm.get("videos"),
            audios=mm.get("audios"), mm_processor_kwargs=mmk)
        if self.opsd:
            return await self._run_opsd(sampling_params, messages, mm, mmk, prompt_ids, **kwargs)
        if self.cgrpo and int(kwargs.get("cgrpo_slot", -1)) >= 0:
            return await self._run_cgrpo(sampling_params, messages, mm, mmk, prompt_ids, **kwargs)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            out: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids,
                sampling_params=sampling_params, mm_processor_kwargs=mmk)
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        resp = out.token_ids[:rlen]
        stats = {"critic_attempted": 0, "critic_parsed": 0, "repaired": 0}

        gt = None
        rm = kwargs.get("reward_model")
        if isinstance(rm, dict): gt = rm.get("ground_truth")

        # Finished: grade exactly against gt and intervene only if wrong (more accurate than having the teacher judge, and free).
        # Unfinished (hit the length cap): there is no answer to grade. The original design skipped these; in the GT variant they go to the teacher,
        # which holds the correct answer and judges whether this path can still succeed; if it sees no problem it replies NONE and we do not intervene.
        capped = len(out.token_ids) >= rlen
        text = self.tokenizer.decode(resp, skip_special_tokens=True)
        ready = self.enabled and self.teacher_manager is not None and gt is not None
        is_correct = False
        if capped:
            need = ready and self.give_gt
        else:
            is_correct = bool(ready) and await self._grade_async(text, gt)
            need = ready and (not is_correct or self.critic_all)
            if is_correct and need:
                stats["critic_on_correct"] = 1
                self._tally(correct_critiqued=1)

        if self.drop_wrong and (capped or not is_correct):
            stats["dropped_wrong"] = 1
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
            return AgentLoopOutput(
                prompt_ids=prompt_ids, response_ids=resp, response_mask=[0] * len(resp),
                response_logprobs=out.log_probs[:rlen] if out.log_probs is not None else None,
                multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2, metrics=metrics,
                extra_fields={**(out.extra_fields or {}), "critic_opd": stats})
        if capped:
            self._tally(capped=1)
        elif not need:
            self._tally(correct=1)
        if need and self.anneal is not None:
            _b = self._beta()
            stats["anneal_beta"] = _b
            if random.random() >= _b:
                need = False
                stats["anneal_skip"] = 1
                self._tally(anneal_skip=1)

        if need:
            self._tally(attempted=1)
            stats["critic_attempted"] = 1
            segs = segment(text)
            body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
            ref = self._reference(messages[-1]["content"]) if (self.give_gt and not self.kerr) else None
            cids = None
            if ref is not None:
                cmsg = [{"role": "system", "content": CRITIC_SYS_REF_CAPPED if capped else CRITIC_SYS_REF},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         "Reference solution (correct; the student may use a different "
                         f"valid approach):\n{ref}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
                cids = self.tokenizer.encode(self.tokenizer.apply_chat_template(
                    cmsg, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
                if len(cids) + self.critic_max_tokens > self.critic_maxlen:
                    cids = None                     # does not fit with the reference solution: fall back to the critique without it
                    stats["critic_ref_overflow"] = 1
            stats["critic_ref"] = int(cids is not None)
            if self.ref_file:
                self._tally(**({"ref_used": 1} if cids is not None else {"ref_missing": 1}))
            if cids is None and self.kerr:
                cmsg = [{"role": "system", "content": CRITIC_SYS_MULTI_CAPPED if capped else (CRITIC_SYS_MULTI_ANY if is_correct else CRITIC_SYS_MULTI)},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
            elif cids is None and self.give_gt:
                cmsg = [{"role": "system", "content": CRITIC_SYS_GT_CAPPED if capped else CRITIC_SYS_GT},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
            elif cids is None:
                cmsg = [{"role": "system", "content": CRITIC_SYS},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\nStudent's solution, numbered segments:\n\n{body}"}]
            if cids is None:
                cids = self.tokenizer.apply_chat_template(cmsg, add_generation_prompt=True, tokenize=False)
                cids = self.tokenizer.encode(cids, add_special_tokens=False)
            _cmax = max(1024, self.critic_max_tokens) if (self.kerr and not self.critic_tokens_exact) else self.critic_max_tokens
            ctok = await self._teacher_generate(cids, _cmax)
            if ctok:
                ctext = self.tokenizer.decode(ctok, skip_special_tokens=True)
                if self.kerr and len(ctok) >= _cmax:
                    # The multi-error critique was truncated: the last feedback entry is unfinished and the student would continue from half a sentence -- drop this block
                    # and cut at the previous complete error (offline: 25.6% truncated with a 1024 cap, 5.4% with 2048)
                    ctext = drop_incomplete_tail(ctext)
                    stats["critic_truncated"] = 1
                    self._tally(kerr_trunc_drop=1)
                if self.kerr:
                    errs = parse_critic_multi(ctext, segs)
                    n_raw = len(errs)
                    _first_raw = errs[0][0] if errs else None   # boundary for the prefix loss: the first error the critique found (including ones removed by the leak filter)
                    if not self.fb_answer:
                        errs = drop_leaky(errs, gt, text)
                    if n_raw > len(errs):
                        stats["critic_n_leak"] = n_raw - len(errs)
                        self._tally(kerr_leak_drop=n_raw - len(errs))
                    sel = kerr_select(errs, self.kerr)
                    idx, fb = (sel[-1][0], kerr_feedback(sel)) if sel else (None, None)
                    first_idx = _first_raw if sel else None
                    stats["critic_n_err"] = len(errs); stats["critic_k_eff"] = len(sel)
                    if sel:
                        self._tally(**{f"kerr_k{len(sel)}": 1})
                else:
                    idx, fb = parse_critic(ctext, segs)
                    first_idx = idx
                if idx is not None and fb:
                    stats["critic_parsed"] = 1
                    self._tally(parsed=1)
                    # The k ablation always keeps the erroneous segment: otherwise the first k-1 errors stay in the context but the k-th is deleted, and the feedback no longer makes sense
                    keep = segs[:idx] if (self.splice == "before" and not self.kerr) else segs[:idx + 1]
                    if self.cut_mode == "end" and not capped:
                        keep = segs
                        stats["cut_at_end"] = 1
                    if self.fb_mode == "generic":
                        fb = GENERIC_FEEDBACK
                    if self.fb_mode == "answer":       # ablation: the feedback states only the reference answer
                        fb = f"The correct final answer is {gt}."
                        stats["fb_answer"] = 1
                    elif self.fb_answer:
                        fb = f"{fb}\nThe correct final answer is {gt}."
                        stats["fb_answer"] = 1
                    kept_txt = "\n\n".join(keep)
                    _k = _tok_prefix_for_text(self.tokenizer, resp, kept_txt)
                    kept_ids = resp[:_k]
                    # The rewrite context must match the tokens actually kept exactly, so decode it from kept_ids
                    # rather than using kept_txt (the two can differ by a few whitespace characters).
                    kept_txt = self.tokenizer.decode(kept_ids, skip_special_tokens=True)
                    ctx = kept_txt + (FEEDBACK_TMPL.format(fb=fb) if self.use_feedback else "\n\n")
                    ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
                    budget = max(256, rlen - len(kept_ids))
                    if self.trunc:
                        budget = min(budget, self.trunc_max)
                    if self.rewriter == "teacher":
                        # The teacher writes the rewrite: no student sampling logprob exists; it is used only for monitoring metrics (with bypass_mode=false and
                        # rollout_is/rs disabled, old_log_probs are recomputed by the actor), so fill with 0
                        with simple_timer("repair", metrics):
                            _tids = await self._teacher_generate(
                                prompt_ids + ctx_ids, budget,
                                temperature=sampling_params.get("temperature", 1.0), top_p=sampling_params.get("top_p", 1.0))
                        r = out
                        _rep_ids = list(_tids or [])
                        _rep_lp = None
                        stats["teacher_rewrite"] = 1
                    else:
                        with simple_timer("repair", metrics):
                            r: TokenOutput = await self.server_manager.generate(
                                request_id=uuid4().hex, prompt_ids=prompt_ids + ctx_ids,
                                sampling_params={**sampling_params, "max_tokens": budget},
                                mm_processor_kwargs=mmk)
                        _rep_ids = list(r.token_ids)
                        _rep_lp = list(r.log_probs) if r.log_probs is not None else None
                    if self.trunc:
                        _kc = _cut_at_blank_line(self.tokenizer, _rep_ids)
                        _rep_ids = _rep_ids[:_kc]
                        if _rep_lp is not None:
                            _rep_lp = _rep_lp[:_kc]
                        stats["repair_tokens"] = _kc
                    # Training sequence = kept prefix (mask 0) + student rewrite (mask 1). The feedback is not in the sequence.
                    _resp_r = (kept_ids + _rep_ids)[:rlen]
                    _npre = 0
                    if self.prefix_loss and first_idx:
                        _npre = min(_tok_prefix_for_text(self.tokenizer, resp, "\n\n".join(segs[:first_idx])), len(kept_ids))
                        stats["prefix_loss_tokens"] = _npre
                    _mask_r = ([1] * _npre + [0] * (len(kept_ids) - _npre) + [1] * len(_rep_ids))[:rlen]
                    # Keep only correct rewrites: if not fixed (including hitting the length cap or not writing an answer), do not return the repair result;
                    # fall through to the end of the function and train the original rollout with plain OPD (num_turns=2).
                    _accept = True
                    if self.keep_correct:
                        _accept = await self._verify_async(
                            self.tokenizer.decode(_resp_r, skip_special_tokens=True), gt)
                        stats["repair_verified"] = int(_accept)
                        self._tally(**({"repair_ok": 1} if _accept else {"repair_bad": 1}))
                    if _accept:
                        resp, mask = _resp_r, _mask_r
                        stats["repaired"] = 1
                        type(self)._CNT["repaired"] += 1
                        metrics.setdefault("num_preempted", -1)
                        # If response_logprobs is missing, _postprocess hits None in
                        #   torch.cat([input.response_logprobs for input in inputs])
                        # (unrepaired samples have it, repaired ones do not). The prefix uses the original rollout's
                        # logprobs (_k counts original tokens, so they align exactly); the rewrite uses its own.
                        _lp = None
                        if out.log_probs is not None:
                            _rlp = _rep_lp if _rep_lp is not None else [0.0] * len(_rep_ids)
                            _lp = (list(out.log_probs[:_k]) + _rlp)[:len(resp)]
                            if len(_lp) < len(resp):          # length must match response_ids
                                _lp += [0.0] * (len(resp) - len(_lp))
                        # L_fb (R4): the teacher must rescore the repair tokens in the context "with feedback",
                        # while the student is trained on the sequence "without feedback" -- this is the strict form of internalizing the correction.
                        # Here we only pass out what the rescoring needs; the actual teacher forward pass is done in
                        # AgentLoopWorker._compute_teacher_logprobs_single_sample
                        # (which already holds the teacher_server_manager and the routing key).
                        # The number of repair tokens must use the actual length after truncation, not len(r.token_ids).
                        _fb = {}
                        if os.environ.get("CRITIC_OPD_LOSS") == "fb":
                            _nrep = max(0, len(resp) - len(kept_ids))
                            if _nrep > 0:
                                _fb = {"critic_fb_prompt_ids": prompt_ids + ctx_ids,
                                       "critic_fb_n_kept": len(kept_ids),
                                       "critic_fb_n_repair": _nrep}
                        if os.environ.get("CRITIC_OPD_DUMP_DIR"):
                            _orig = list(out.token_ids[:rlen])
                            _fe = (_tok_prefix_for_text(self.tokenizer, _orig, "\n\n".join(segs[:first_idx]))
                                   if first_idx is not None else None)
                            self._dump({"uid": str(kwargs.get("uid")), "gt": str(gt), "prompt_ids": list(prompt_ids),
                                        "orig_ids": _orig, "correct": bool(is_correct), "capped": bool(capped),
                                        "repaired": 1, "kept_len": len(kept_ids), "first_err_tok": _fe, "fb": fb,
                                        "ctx_ids": list(ctx_ids), "rep_ids": list(resp[len(kept_ids):])})
                        return AgentLoopOutput(
                            prompt_ids=prompt_ids, response_ids=resp, response_mask=mask,
                            response_logprobs=_lp,
                            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=4,
                            metrics=metrics, extra_fields={**(r.extra_fields or {}),
                                                           "critic_opd": stats,
                                                           **_fb})

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
        if os.environ.get("CRITIC_OPD_DUMP_DIR"):
            self._dump({"uid": str(kwargs.get("uid")), "gt": str(gt), "prompt_ids": list(prompt_ids),
                        "orig_ids": list(resp), "correct": bool(is_correct), "capped": bool(capped), "repaired": 0})
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
            response_logprobs=out.log_probs[:rlen] if out.log_probs is not None else None,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2, metrics=metrics,
            extra_fields={**(out.extra_fields or {}), "critic_opd": stats})
