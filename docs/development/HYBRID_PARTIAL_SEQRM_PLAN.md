# Hybrid-memory partial `seq_rm` for speculative decoding

**Status:** design, not yet implemented
**Author of this plan:** handoff doc; read before implementing
**Goal:** make spec-decoding actually accelerate hybrid attention+SSM models (Qwen3.5/3.6, Qwen3-Next, Mamba variants)

---

## 1. The problem

Speculative decoding works by:

1. Target decodes a batch `[last_accepted, d1, …, dK]` in one forward pass (K+1 positions).
2. Rejection-samples each draft. Accepts the first `M ≤ K` drafts; resamples at the rejection point.
3. Next iteration: continue from after the last accepted token.

For step 3 to be cheap, the context has to be able to **roll back** the last `K - M` positions' worth of state. For plain attention, this is a pointer move: `num_computed_tokens -= K - M`; the orphaned KV blocks get overwritten on the next write. Zero work.

For **SSM / linear-attention / GDN / Mamba** layers, state is recurrent. The "state after K+1 tokens" is a function of all K+1 tokens. You cannot undo the last step by pointer arithmetic. You either need:

- A snapshot you can restore to, **or**
- The per-token intermediate states, so you can pick slot[M].

llama.cpp's `llama_memory_recurrent` (`src/llama-memory-recurrent.*`) today supports **neither**. `seq_rm(seq, p0, -1)` with `p1 == -1` is the only form it accepts, and for recurrent memory that's effectively "clear the whole sequence." Any mid-sequence truncation fails, which makes `common_context_can_seq_rm()` return `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`.

The server's fallback when it sees `TYPE_FULL` is the **checkpoint path** (`tools/server/server-context.cpp`, see `spec_ckpt` / `server_get_checkpoint` / `llama_state_seq_set_data_ext`):

1. Save full per-sequence state to host memory (a dump of the KV + SSM + conv state).
2. Do the verify batch.
3. On reject: restore the full state, set `spec_draft = [target's resampled token]`, and **loop the verify** — doing *another* batch=K+1 target decode that trivially accepts (since "draft" is now target's own sample).

Cost per reject: **two** batch=K+1 target decodes instead of one. On Qwen3.5's hybrid-SSM architecture where batch=2 takes roughly 2× batch=1 (SSM layers don't parallelize inside a sequence), this doubles the reject path cost and almost entirely cancels the spec-decode speedup:

```
baseline (no spec):  4.80 tok/s
K=1:                 4.87 tok/s  (+1%)   ← theoretical 1.67×, achieved ~1.01×
K=2:                 5.26 tok/s  (+10%)
K=3:                 5.10 tok/s  (+6%)
K=4:                 4.63 tok/s  (-4%)
```

---

## 2. What vLLM does, in one paragraph

For each drafter-equipped request, vLLM allocates **`1 + num_speculative_tokens` state slots per SSM/linear-attention layer**, not one. During the target's single batched forward over the K+1 positions, the SSM / `causal_conv1d` / GDN kernels write **every intermediate per-token state into its own slot** (not overwriting a single slot). After the batched decode, sampling produces `num_accepted_tokens`. The **next** step's kernel reads its initial state from `slot[num_accepted - 1]` — no memcpy, no re-forward. Rollback is a single index load.

References (vLLM main as of this plan's date):

- Slot allocation: `vllm/v1/attention/backends/mamba_attn.py` (`self.state_indices_tensor_d = torch.empty((..., 1 + self.num_spec_tokens), ...)`)
- Kernel-level indexed read: `vllm/model_executor/layers/mamba/ops/mamba_ssm.py`, look for `num_accepted_tokens_ptr` and `init_token_idx = max(num_accepted - 1, 0)`.
- Conv state equivalent: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`.
- Linear-attention / GDN (what Qwen3.5 uses): `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `fused_sigmoid_gating_delta_rule_update(..., num_accepted_tokens=…)`.
- Plain-attention rollback (pointer move): `vllm/v1/core/sched/scheduler.py`, `request.num_computed_tokens -= num_rejected`.

vLLM's mechanism for SSM rollback is **select, not restore**. No per-token snapshot; the snapshots are produced by the scan itself.

---

## 3. The simplifying observation about our kernels

In llama.cpp, `ggml_ssm_scan` and `ggml_ssm_conv` *already compute per-token intermediate states internally* (they have to — it's a scan). They just don't return them. The work to expose them as output tensors is layout plumbing, not new math. Same for the linear-attention ops used by Qwen3.5 (`build_layer_attn_linear` in `src/models/qwen35.cpp`, `build_delta_net` from `llm_build_delta_net_base`).

This keeps the kernel change scoped to "emit an additional tensor" rather than anything algorithmic.

---

## 4. Proposed implementation, staged

### Stage 1 — snapshot+restore for all-or-nothing rollback *(~1 week)*

**Goal:** for the K=1 spec path (and any K>1 case where either all-accepted or all-rejected), eliminate the checkpoint-reverify loop. Accept that partial K>1 rejects still use the existing checkpoint path for now.

Changes:

| file | change |
|---|---|
| `src/llama-memory-recurrent.h/.cpp` | Add `snapshot(seq_id) → snapshot_handle` and `restore(seq_id, snapshot_handle)`. Implemented as a device-to-device `ggml_backend_tensor_copy` of the sequence's `r_l`/`s_l` slot contents into a scratch buffer owned by the memory module. Small (Qwen3.5 ≈ 75 MB per snapshot). |
| `src/llama-memory-hybrid.h/.cpp` | Wire snapshot/restore through: the hybrid memory snapshots the recurrent half and lets the attention half handle itself via existing partial `seq_rm`. |
| `src/llama-memory.h` | Add virtual `snapshot`/`restore` (no-op defaults) and a `has_fast_partial_seq_rm()` flag. |
| `common/common.cpp` | `common_context_can_seq_rm` returns `TYPE_PART` when `has_fast_partial_seq_rm()` is true for hybrid memory. |
| `tools/server/server-context.cpp` | When `ctx_seq_rm_type == TYPE_PART`, skip the checkpoint path — use `llama_memory_seq_rm` directly. |

Verification:
- `./build/bin/test_mtp_rollout` should still show ~66% t+2 agreement.
- `llama-server --spec-type mtp --draft-max 1` should show **no checkpoint restore line** in the server log on rejects.
- Expected speedup: move from current +10% (K=2) to roughly +15–25%, depending on reject frequency.

Risks:
- Snapshot cost on device is non-trivial (~75 MB GPU→GPU copy ≈ 5–10 ms). If we snapshot before *every* spec verify, overhead compounds. Mitigation: snapshot **lazily** — only when the spec path is actually going to run (already gated by `slot.can_speculate()`).
- The snapshot has to include the *conv* state, not just `r_l`/`s_l`. Qwen3.5 stores it in `ssm_conv1d` state; confirm the full state set (see `llama_memory_recurrent::state_write_data`).

### Stage 2 — multi-slot per-token state during batched decode *(~2 weeks)*

**Goal:** match vLLM's cost model. Partial K>1 rejects become cheap. Eliminates checkpoint path entirely for hybrid models.

This is where the kernel work lives. Breakdown:

#### 2a. Memory-module layout change

Extend `llama_memory_recurrent` to hold `1 + n_draft_max` state slots per sequence, selectable by a per-sequence "active slot" index. Current per-layer tensors (`r_l[il]`, `s_l[il]`) are shape `[state_dim, n_seq_max, n_tokens_slot]`; the new axis is the slot.

Public API (on `llama_memory_recurrent`):
```cpp
void set_active_slot(llama_seq_id seq, int32_t slot);  // 0..n_draft_max
int32_t get_active_slot(llama_seq_id seq) const;
void commit_slot(llama_seq_id seq, int32_t slot);  // promote slot → 0
```

#### 2b. ggml op variants that emit per-token intermediate states

The scan ops need a mode where, instead of returning `[y, final_state]`, they return `[y, state_0, state_1, ..., state_{T-1}]`. The CUDA kernels in `ggml/src/ggml-cuda/ssm-scan.cu` and `ssm-conv.cu` already compute these values per step; they just need an additional output buffer pointer and a write loop.

Scope:
- `ggml_ssm_scan` + backend kernels (CPU, CUDA, Metal, HIP as applicable) — the Mamba path.
- Linear-attention ops for Qwen3-family models. `build_delta_net` in `llm_build_delta_net_base` uses standard ggml mul-mats + `ggml_ssm_scan` inside — so this mostly reduces to the scan op change.
- `ggml_ssm_conv` (or equivalent) — the causal conv 1-D state.

Each of these needs a new "emit intermediates" variant. The simpler path is to add a variant tensor (`s_intermediates` with shape `[state_dim, n_tokens, n_seq]`) alongside the existing state output. The original path (final-state-only) stays for non-spec users.

#### 2c. Graph builder wiring

For Qwen3.5 specifically (`src/models/qwen35.cpp::build_layer_attn_linear`):
- Replace the single-state update with the "emit intermediates" op.
- Route the per-token intermediate states to `recurrent_memory.slots[0..K]` via `ggml_cpy` (on-device, no host copy).

#### 2d. Plumbing `num_accepted` from sampler → next graph

A new per-sequence integer input to the graph: `num_accepted`. Before each decode, the sampler writes how many drafts from the previous step were accepted. The linear-attention / SSM graph reads its initial state from `slots[num_accepted - 1]` (or `slots[0]` if `num_accepted == 0`).

This mirrors vLLM's `num_accepted_tokens_ptr` exactly.

#### 2e. Speculative verify glue

`common_speculative_state_mtp::accept(n_accepted)` currently a no-op. It should set the active slot index on the context so the next `llama_decode` picks up the right state.

Same API surface as vLLM; just a small int that propagates into the graph input.

Verification:
- Acceptance metrics unchanged (they weren't wrong; the counter artifact is on the server side, not in our state).
- Eliminate the checkpoint path entirely for hybrid memory in `server-context.cpp` — set `ctx_seq_rm_type = TYPE_PART` unconditionally for hybrid.
- Expected speedup: approach the theoretical 1.67× that vLLM shows on pure-attention MTP, modulo Qwen3.5's SSM-layer batch inefficiency.

---

## 5. Out of scope for this plan

- **Kernel parallelism for SSM layers across batch positions.** Even with the rollback fix, Qwen3.5's per-token SSM cost still dominates; batch=2 is about 2× batch=1 wall-time on our hardware. That's a separate optimization (fused GDN kernels, etc.) and affects baseline, not just spec.
- **D > 1 MTP.** All of this works for D=1 (the only deployed MTP model today). If a D>1 model ships, the chain mechanics in `common_speculative_state_mtp::draft` change slightly but the memory layer work above is unaffected.
- **EAGLE3 support.** The existing EAGLE3 stub in `common/speculative.cpp:553` is orthogonal. Stage 1 & 2 benefit any spec type that runs on a hybrid-memory model, not just MTP.

---

## 6. Recommended implementation order

1. Stage 1 end-to-end first, even if modest speedup — it **proves the plumbing** (memory-module snapshot APIs, the `has_fast_partial_seq_rm` flag, server side-path). Stage 2 reuses all of it.
2. Within Stage 2, start with `ggml_ssm_scan` + CPU backend (easy to debug), then CUDA, then the other backends that actually matter (Metal, HIP).
3. Linear-attention path for Qwen3.5 is its own bundle — do the pure-Mamba kernel first, then port the pattern.
4. Only remove the server checkpoint path after Stage 2 is validated; it stays as a fallback path for any model that doesn't opt into multi-slot memory.

---

## 7. How to validate at each stage

**Correctness probe:** keep `scripts/test_mtp_rollout.cpp`. t+2 rate is architecture-correct regardless of rollback mechanism; if it drops, you broke the state handling.

**Throughput probe:**
```
./build/bin/llama-server -m ../qwen3.6-bf16-mtp.gguf -c 4096 \
    --spec-type mtp --draft-max 2 --draft-p-min 0 --no-warmup
curl -s http://127.0.0.1:8080/completion -d '{"prompt":"Once upon a time in a distant galaxy,","n_predict":200,"temperature":0,"cache_prompt":false}' | jq .timings
```
Baseline for this branch today: 5.26 tok/s at K=2. Stage 1 target: 5.6+ tok/s. Stage 2 target: 7+ tok/s (approaches 1.6× baseline).

**Spec-verify tracing:** the `SPEC_VERIFY_DEBUG=1` probe I added and reverted (`common/sampling.cpp::common_sampler_sample_and_accept_n`) can be re-added as a temporary tool. Real-accept vs loopback-accept is the signal.

---

## 8. Files that will be touched, for a handoff check-in

Stage 1:
- `src/llama-memory.h`
- `src/llama-memory-recurrent.h/.cpp`
- `src/llama-memory-hybrid.h/.cpp`
- `common/common.cpp` (`common_context_can_seq_rm`)
- `tools/server/server-context.cpp` (skip checkpoint when TYPE_PART)

Stage 2:
- Everything above, plus:
- `ggml/src/ggml.c` (new `ggml_ssm_scan_emit_states` op)
- `ggml/src/ggml-cuda/ssm-scan.cu` (+ `ssm-conv.cu`)
- `ggml/src/ggml-cpu/ops/ssm-scan.cpp` (backend CPU fallback)
- `src/models/qwen35.cpp::build_layer_attn_linear` (use emit-states op + slot routing)
- `src/llama-graph.h/.cpp` (plumb `num_accepted` input)
- `common/speculative.cpp::common_speculative_state_mtp::accept` (set active slot)

Total LOC estimate:
- Stage 1: ~400
- Stage 2: ~1500 including backend variants (CPU/CUDA at minimum)

---

## 9. Starting point when picking this up

The current branch at `2a070609a` has:
- K=1 and K-chained MTP working end-to-end via `llama_mtp_decode`.
- Server integration via `common_speculative_state_mtp` and `--spec-type mtp`.
- K=2 demonstrates +10% end-to-end speedup on Qwen3.6, limited by exactly the rollback problem this plan addresses.
- `scripts/test_mtp_rollout.cpp` for regression testing the draft semantics.

Read `common/speculative.cpp::common_speculative_state_mtp` and `src/models/qwen35.cpp::build_standalone_mtp` for the current flow. Then pick up Stage 1 with `src/llama-memory-recurrent.{h,cpp}` — the `snapshot` / `restore` methods are the smallest committable unit.
