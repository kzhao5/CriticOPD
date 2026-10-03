"""Trajectory-level occupancy-term steering (轨迹级占据项 steering), faithful to the
OPD-yyx design doc. Adds a second term on the samplek/OPD trunk: each trajectory gets a
scalar advantage A_hat, broadcast to all its tokens.

  ell[g,t] = per-token divergence (larger = more deviation from teacher), * response_mask
  S[g]     = sum_t ell[g,t]                          (trajectory total divergence; SUM not mean)
  b[g]     = mean_{g' != g, same prompt} S[g']       (leave-one-out group baseline)
  d[g]     = S[g] - b[g]
  scale    = EMA_0.99( median_g |d[g]| * 1.4826 )    (cross-step MAD scale)
  A_hat[g] = clip( -d[g] / scale, -5, +5 )           (NEGATIVE: high divergence -> suppressed)
  A_total  = A_base + beta * A_hat[g] * response_mask
beta: PI controller so ||beta*steer|| / ||base|| ~= grad_ratio_target, clamped [1e-4, 0.3].
Caller handles warmup (first 20 steps beta=0). State (ema_scale, beta) persists on caller.
Hard reqs: no mean; group baseline MUST be leave-one-out; do NOT divide by group std.
"""
import numpy as np
import torch

EPS = 1e-8


def compute_traj_steer(ell, response_mask, uid, state, grad_ratio_target=0.1,
                       mode="sum", theta_q=0.9, clip=5.0):
    """ell,response_mask (B,T); uid (B,) int group id; state dict persisted by caller.
    Returns steer_tok (B,T) = A_hat[g] broadcast * mask (detached, PRE-beta), beta, stats."""
    ell = ell * response_mask
    B, T = ell.shape
    if mode == "event":
        # variant: S[g] = count of tokens with ell > global running q90 (EMA 0.99)
        vals = ell[response_mask > 0]
        cur_q = torch.quantile(vals, theta_q).item() if vals.numel() > 0 else 0.0
        state["ema_q"] = 0.99 * state.get("ema_q", cur_q) + 0.01 * cur_q
        S = ((ell > state["ema_q"]).float() * response_mask).sum(-1)
    else:
        S = ell.sum(-1)  # trajectory total divergence (sum)

    # leave-one-out group baseline b[g], d[g] = S[g] - b[g]
    d = torch.zeros(B, device=ell.device)
    if not torch.is_tensor(uid):
        uid = torch.as_tensor(uid, device=ell.device)
    for u in torch.unique(uid):
        idx = (uid == u).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n >= 2:
            Sg = S[idx]
            b = (Sg.sum() - Sg) / (n - 1)   # leave-one-TRAJECTORY-out
            d[idx] = Sg - b
        # single-trajectory group -> d=0 (no steering, can't LOO)

    # cross-step MAD scale
    cur_mad = torch.median(torch.abs(d)).item() if B > 0 else 1.0
    state["ema_scale"] = 0.99 * state.get("ema_scale", cur_mad) + 0.01 * max(cur_mad, 1e-6)
    scale = 1.4826 * state["ema_scale"] + EPS
    A_hat = torch.clamp(-d / scale, -clip, clip)        # (B,)  negative sign

    steer_tok = A_hat.view(B, 1) * response_mask         # broadcast to tokens (PRE-beta)

    # beta PI: ||beta*steer|| / ||base|| ~ grad_ratio_target  (base proxy = ||ell||)
    base_norm = ell.norm().item()
    steer_norm = steer_tok.norm().item()
    beta_raw = grad_ratio_target * base_norm / (steer_norm + EPS)
    beta = float(np.clip(beta_raw, 1e-4, 0.3))
    state["beta"] = 0.9 * state.get("beta", beta) + 0.1 * beta
    beta = state["beta"]

    stats = dict(
        steer_beta=beta,
        steer_S_mean=float(S.mean().item()),
        steer_Ahat_mean=float(A_hat.mean().item()),
        steer_neg_frac=float((A_hat < 0).float().mean().item()),
        steer_scale=float(scale),
    )
    return steer_tok.detach(), beta, stats
