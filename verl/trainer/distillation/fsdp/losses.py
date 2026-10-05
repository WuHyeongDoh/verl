# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _chunked_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute log_softmax(logits).gather(topk_ids) without materializing [B, T, V].

    Uses the identity:
        log_softmax(x).gather(idx) == x.gather(idx) - logsumexp(x, keepdim=True)
    Streams the reduction in chunks of `chunk_size` tokens along (B*T) with fp32
    logsumexp for numerical stability.

    Args:
        logits:    [B, T, V] student logits.
        topk_ids:  [B, T, K] indices to gather.
        chunk_size: number of tokens per chunk; only affects memory, not numerics.

    Returns:
        [B, T, K] tensor with the same dtype as `logits`.
    """
    B, T, V = logits.shape
    K = topk_ids.shape[-1]
    flat_logits = logits.reshape(-1, V)  # [N, V]
    flat_topk = topk_ids.reshape(-1, K)  # [N, K]
    N = flat_logits.shape[0]

    # Edge case: empty input (e.g. fully-padded micro-batch).
    if N == 0:
        return torch.empty((B, T, K), dtype=logits.dtype, device=logits.device)

    out = torch.empty((N, K), dtype=logits.dtype, device=logits.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        chunk_logits_fp32 = flat_logits[s:e].float()
        log_z = torch.logsumexp(chunk_logits_fp32, dim=-1, keepdim=True)  # [c, 1]
        chunk_topk_logits = torch.gather(chunk_logits_fp32, dim=-1, index=flat_topk[s:e])
        out[s:e] = (chunk_topk_logits - log_z).to(logits.dtype)
    return out.reshape(B, T, K)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    # ``use_chunked_topk`` (opt-in, default off) trades latency for memory:
    # the chunked path streams logsumexp + gather to avoid the [B, T, V]
    # log_softmax buffer, enabling long-context (>=64K) where the default
    # F.log_softmax path OOMs. See ``DistillationLossConfig.use_chunked_topk``
    # for trade-offs and benchmark numbers.
    loss_config: DistillationLossConfig = config.distillation_loss
    use_chunked_topk = getattr(loss_config, "use_chunked_topk", False)
    if use_chunked_topk:
        # log_softmax is monotonic, so topk(logits) == topk(log_softmax(logits)).
        student_topk_ids = torch.topk(student_logits, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = _chunked_topk_log_probs(
            student_logits,
            teacher_topk_ids,
            chunk_size=getattr(loss_config, "chunked_topk_chunk_size", 4096),
        )
    else:
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
        student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


# ---------------------------------------------------------------------------------------------------------------------
# [rr-opd] Hybrid distillation terms: sampled reverse-KL (policy gradient) on most tokens, top-K forward KL on gated
# tokens.  Teacher tensors use the layout  [actual | top-1 .. top-K | aux]  (see extract_prompt_logprobs with
# RR_TOPK_WITH_ACTUAL=1 and AgentLoopWorker._compute_teacher_logprobs with RR_AUX_COL=1):
#   column 0      : id / teacher logprob of the token that is in the sequence (the sampled token)
#   columns 1..K  : the teacher's top-K ids / logprobs at this prefix
#   column K+1    : logprob slot = row marker (0 prompt row, 1 student-sampled response token, 2 teacher-sampled prefix)
# Everything is computed on response rows only (fp32), so the memory cost scales with the response length.
# Env knobs (read once per call; all optional):
#   RR_GATE            off | meta | support          (default off)
#   RR_META_TOKENS     path to a JSON list of token ids = the meta-cognitive lexicon (gate = meta)
#   RR_GATE_TAU        teacher mass threshold  (default 0.5)
#   RR_GATE_EPS        student mass threshold  (default 0.05)
_RR_META_CACHE: dict = {}


def _rr_meta_ids(device) -> torch.Tensor | None:
    import json
    import os

    path = os.environ.get("RR_META_TOKENS", "")
    if not path:
        return None
    key = (path, str(device))
    if key not in _RR_META_CACHE:
        ids = json.load(open(path))
        ids = ids["ids"] if isinstance(ids, dict) else ids
        _RR_META_CACHE[key] = torch.tensor(sorted(set(int(i) for i in ids)), dtype=torch.long, device=device)
    return _RR_META_CACHE[key]


def compute_rr_hybrid_terms(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Per-token terms for the hybrid loss.  All returned tensors have shape (1, total_nnz); only ``rr_fkl`` carries grad.

    rr_fkl        KL( p~_T || p_theta ) with the teacher renormalised over its top-K and the student's FULL-vocabulary
                  log-probabilities (so minimising it moves absolute student mass onto the teacher's tokens); the value is
                  filled on every response row, the gradient only on rows whose forward KL enters the loss (gated rows,
                  and teacher-prefix rows when RR_PREFIX_LOSS=fkl)
    rr_teacher_lp teacher logprob of the sampled token (for the k1 / reverse-KL policy-gradient term)
    rr_gate       1 where the forward-KL term replaces the policy-gradient term (gate), else 0
    rr_prefix     1 where the token was sampled by the teacher (teacher-prefix rollout), else 0
    rr_resp       1 on response rows
    rr_meta_t / rr_meta_s   teacher / student mass on the gate's token set (diagnostics)
    teacher_mass / student_mass   mass inside the teacher's top-K
    """
    import os

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    assert get_ulysses_sequence_parallel_world_size() == 1, "rr_hybrid does not support Ulysses sequence parallelism"
    t_lp_all = teacher_topk_log_probs.values()  # (nnz, K+2)
    t_ids_all = teacher_topk_ids.values()
    logits = student_logits.squeeze(0)  # (nnz, V)
    nnz = logits.shape[0]
    assert t_lp_all.shape[0] == t_ids_all.shape[0] == nnz, (t_lp_all.shape, t_ids_all.shape, logits.shape)
    K = t_lp_all.shape[-1] - 2
    assert K >= 1, f"rr_hybrid expects teacher tensors [actual | top-K | aux], got width {t_lp_all.shape[-1]}"

    aux = t_lp_all[:, K + 1]
    resp = aux > 0.5
    idx = resp.nonzero(as_tuple=True)[0]
    dev = logits.device
    z = torch.zeros(nnz, dtype=torch.float32, device=dev)
    out = {k: z.clone() for k in ("rr_teacher_lp", "rr_gate", "rr_prefix", "rr_resp", "rr_meta_t", "rr_meta_s", "teacher_mass", "student_mass")}
    out["rr_resp"] = resp.float()
    out["rr_prefix"] = (aux > 1.5).float()
    out["rr_teacher_lp"] = t_lp_all[:, 0].float()
    fkl_full = z.clone()
    if idx.numel() > 0:
        t_lp = t_lp_all[idx, 1 : K + 1].float()  # (R, K)
        t_ids = t_ids_all[idx, 1 : K + 1].long()
        p_t = t_lp.exp()
        t_mass = p_t.sum(dim=-1)
        p_tn = p_t / t_mass.clamp_min(1e-8).unsqueeze(-1)
        log_p_tn = p_tn.clamp_min(1e-12).log()
        chunk = int(os.environ.get("RR_ROW_CHUNK", "1024"))

        def student_topk_lp(rows: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            """Student full-vocabulary log-probs at `ids` for logits rows `rows` (fp32, chunked).  index_select copies the
            rows, so this path is independent of the in-place backward of the flash-attn cross-entropy on `logits`."""
            parts = []
            for s0 in range(0, rows.numel(), chunk):
                lg = logits.index_select(0, rows[s0 : s0 + chunk]).float()
                parts.append(torch.gather(lg, dim=-1, index=ids[s0 : s0 + chunk]) - torch.logsumexp(lg, dim=-1, keepdim=True))
            return torch.cat(parts, dim=0)

        # pass 1 (no grad, all response rows): student mass on the teacher's top-K -> gate + diagnostics.
        with torch.no_grad():
            s_lp = student_topk_lp(idx, t_ids)
            p_s = s_lp.exp()
            fkl_ng = (p_tn * (log_p_tn - s_lp)).sum(dim=-1).clamp_min(0.0)
            out["teacher_mass"][idx] = t_mass
            out["student_mass"][idx] = p_s.sum(dim=-1)
            mode = os.environ.get("RR_GATE", "off")
            tau = float(os.environ.get("RR_GATE_TAU", "0.5"))
            eps = float(os.environ.get("RR_GATE_EPS", "0.05"))
            if mode == "meta":
                meta = _rr_meta_ids(dev)
                assert meta is not None, "RR_GATE=meta needs RR_META_TOKENS"
                sel = torch.isin(t_ids, meta)
                m_t, m_s = (p_t * sel).sum(dim=-1), (p_s * sel).sum(dim=-1)
                gate = (m_t >= tau) & (m_s <= eps)
            elif mode == "support":
                sel = p_s < eps  # teacher-preferred tokens the student (almost) never samples
                m_t, m_s = (p_t * sel).sum(dim=-1), (p_s * sel).sum(dim=-1)
                gate = m_t >= tau
            elif mode == "off":
                m_t, m_s = torch.zeros_like(t_mass), torch.zeros_like(t_mass)
                gate = torch.zeros_like(t_mass, dtype=torch.bool)
            else:
                raise ValueError(f"unknown RR_GATE={mode}")
            is_prefix = aux[idx] > 1.5
            gate = gate & ~is_prefix  # the gate applies to student-sampled tokens only
            out["rr_gate"][idx] = gate.float()
            out["rr_meta_t"][idx] = m_t
            out["rr_meta_s"][idx] = m_s
            need = gate | (is_prefix if os.environ.get("RR_PREFIX_LOSS", "ce") == "fkl" else torch.zeros_like(gate))
        fkl_full = fkl_full.index_put((idx,), fkl_ng)
        # pass 2 (with grad): only the rows whose forward KL enters the loss, so memory scales with the gated rows
        sel_rows = need.nonzero(as_tuple=True)[0]
        if sel_rows.numel() > 0:
            s_lp_g = student_topk_lp(idx[sel_rows], t_ids[sel_rows])
            fkl_g = (p_tn[sel_rows] * (log_p_tn[sel_rows] - s_lp_g)).sum(dim=-1).clamp_min(0.0)
            fkl_full = fkl_full.index_put((idx[sel_rows],), fkl_g)
    out["rr_fkl"] = fkl_full
    return {k: v.unsqueeze(0) for k, v in out.items()}
