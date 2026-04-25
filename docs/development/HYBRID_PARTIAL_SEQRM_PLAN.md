# Multi-slot recurrent memory (foundation for hybrid spec decode)

**Status:** plan, not yet implemented. Replaces an earlier patches-on-server design.
**Goal:** give the recurrent half of hybrid memory the structural primitive it
needs for correct, low-cost rollback during speculative decoding —
**(1 + n_draft_max)** state slots per sequence, with kernels that emit
per-token intermediates into those slots during a verify batch. This is the
mechanism vLLM uses for SSM/linear-attention spec decode and the only path
that lets us roll back to "after M accepted drafts" without re-decoding
already-accepted tokens.

Specifically scoped to qwen3.5/3.6 GDN (the linear-attention model we have on
disk and the one the earlier in-tree attempt targeted). Mamba SSM-scan and
RWKV are explicitly out of scope until this lands cleanly for GDN.

---

## 1. Why the slot primitive, not server-side patches

We spent a session chasing two server-layer fixes — a host→device snapshot of
the recurrent state, and a "reverify-elimination" that single-token-redecodes
after restore. Both leave the structural problem untouched: **after a partial
accept of M drafts out of K, the recurrent half is at pos `P + K + 1` and there
is no way to recover the state at pos `P + M + 1` other than re-running the
target.** Every server-layer patch ends up paying that re-run.

vLLM avoids it by allocating `1 + num_speculative_tokens` state slots per
SSM/linear-attention layer per sequence, writing each per-token intermediate
state into its own slot during the batched verify, and reading
`slot[num_accepted - 1]` on the next decode. Rollback is a **select**, not a
restore. No memcpy, no re-forward.

The kernels in `ggml/src/ggml-cuda/ssm-scan.cu`, `ssm-conv.cu`, and the GDN
ops in `ggml/src/ggml-cuda/gated_delta_net.cu` already compute these
intermediates internally — they're produced by the scan. Today they only
return the final state. The work is layout plumbing, not new math.

---

## 2. Why earlier attempts at this failed (read before designing)

`findings.md` in the repo root documents an in-tree attempt that built most of
this and hit an unresolved long-prompt spiral. Key non-trivial findings:

- **Per-slot bit-identity does not imply end-to-end correctness.** The prior
  attempt's `ggml_gated_delta_net_emit` kernel produced slot-N contents that
  were bit-identical to a truncated non-emit run (`scripts/test_gdn_emit.cpp`
  at Qwen3.6 sizes). Yet long-prompt generation collapsed into attractors
  ("people people people…" / "of the of the…") on both CPU and CUDA.
- **The collapse is uniquely triggered when `s_copy` returns a slot index
  > 0** — i.e., when the rollback path actually fires. Widening the tensor
  alone, or running the emit kernel without rollback reads, was coherent.
- **The chunking and AR (autoregressive) GDN kernels live on different
  numerical trajectories** — max-abs-diff 19.65 on activations of order 1
  between Fused-AR and Fused-CH on Qwen3.6 sizes. This is not FP noise.
  Master's normal generation path uses Fused-AR; the verify batch (T > 1)
  uses chunking; emit naturally writes from the chunking path. Every
  partial-accept rollback pulls the seq onto trajectory A while master's
  per-token decode lives on trajectory B. They drift, output collapses.
- **The widened tensor itself perturbed prefill state under `n_parallel > 1`**
  — even with emit disabled and rollback disabled, the mere existence of
  more rows in `r_l`/`s_l` shifted byte-level state during prompt processing.
  Suspected cause: `nb[1]` / find_slot indexing assumptions that hold when
  `mem_size = 1` but break when wider. Unconfirmed.

These three observations bound the plan: any phase that doesn't have a
proven fix for each is not ready to land.

---

## 3. What "slots" looks like concretely

Today, for each recurrent layer:

- `r_l[il]`: shape `[n_embd_r, mem_size]`, where `mem_size = n_seq_max`.
- `s_l[il]`: shape `[n_embd_s, mem_size]`.
- One cell per sequence; cell holds the latest state for that seq.

After:

- `r_l[il]`: shape `[n_embd_r, mem_size * (1 + n_spec_max)]`, flat layout.
- `s_l[il]`: shape `[n_embd_s, mem_size * (1 + n_spec_max)]`.
- Per-cell **slot group** of `1 + n_spec_max` slots. Slot 0 is the active
  committed state. Slots 1..n_spec_max hold per-token intermediates from the
  most recent verify batch.
- Lookup: state for seq `S` at slot `K` lives at flat row `K * mem_size + cells[S].tail`.

Per-sequence active-slot selector (`active_slots[seq]`) controls which slot
the next decode reads from. Defaults to 0 (committed). Set by the spec accept
path on partial accept. **One-shot consume**: after a graph reads `active_slot`,
it is reset to 0 — prevents downstream decodes from re-reading a stale slot
when no rollback fires.

API (on `llama_memory_recurrent`):

```cpp
void   set_active_slot(llama_seq_id seq, int32_t slot);  // 0..n_spec_max
int32_t get_active_slot(llama_seq_id seq) const;
```

Plumbed from `llama_context_params::n_spec_max` set by `common_context_params_to_llama`
when `--spec-type` is given.

---

## 4. The kernel question (must be answered before phase 2)

The whole plan hinges on this and was the failure point of the prior attempt.

The verify batch decodes T = K+1 tokens. The recurrent kernel processes them
as a sequence, internally producing T per-token states. We need to write each
state into slot[t] AND have the resulting slot contents be **trajectory-
compatible** with what the kernel used for normal T=1 decode would produce.

Three options, in order of decreasing risk:

### Option A — chunking emit, with master also forced to chunking

Add the "emit intermediates" output to the chunking GDN op (the prior attempt
already did this). Forces master's normal T=1 generation through the chunking
kernel as well, so trajectories match.

- **Pre-flight gate:** master with `FUSED_GDN_AR=0` (forces trajectory A
  everywhere) must produce coherent output at n_predict=500 on the spiral
  prompts. Per `findings.md`, this was tested and *did* work on the prior
  attempt. Re-verify before committing to this option.
- **Risk:** moves master's hot path from Fused-AR → chunking. May cost
  per-token throughput on T=1 decode that we don't recover from spec wins.
  Needs a benchmark before/after.

### Option B — serialized AR inside the verify batch

In the verify batch's recurrent-layer subgraph, instead of one chunking call
over T tokens, issue T serialized AR-kernel calls. Each writes its post-state
into slot[t]. Attention layers still batch across T (the win comes from there
on hybrid models anyway).

- Same kernel for verify and for normal T=1 generation → same trajectory by
  construction.
- **Risk:** per-T-token serialization eats some of the verify-batch win that
  this plan exists to capture. May still net positive because the alternative
  is "no spec at all" or "snapshot+reverify."
- Implementation surface is higher than option A — must rewrite
  `build_layer_attn_linear` for qwen3.5 to issue the serialized ops and
  thread per-step output into slots.

### Option C — root-cause the AR vs chunking 19.65 divergence

The two kernels should produce numerically equivalent outputs (modulo FP
noise). They don't. One of them is wrong, or there's a mathematical
difference in how chunking handles boundary conditions vs AR.

- If chunking is correct, fix Fused-AR — and option A becomes free.
- If AR is correct, fix chunking emit — and the original "emit-from-chunking"
  approach works without forcing master onto trajectory A.
- **Risk:** unbounded debugging exercise. Could be quick or could be weeks.
  Worth a focused 2-3 day timebox before committing to A or B.

**Recommended sequence:** spend up to 3 days on option C. If unresolved,
switch to option B (the safest correctness story). Keep option A as a fallback
if option B's serialization cost is unacceptable.

---

## 5. Phased plan

### Phase 0 — gates (no code, just regression tests in place)

- `scripts/test_spiral_regression.py` — already exists. Long-prompt
  baseline-vs-spec coherence test with attractor detection.
- `scripts/test_gdn_emit.cpp` — already exists from prior attempt
  (untracked). Per-slot bit-identity check at Qwen3.6 sizes.
- **NEW:** `test_gdn_trajectory.cpp` — directly compare AR-kernel vs
  chunking-kernel trajectories on the same input over N tokens. Asserts
  max-abs-diff < tol on intermediate states. Currently fails by 19.65;
  any phase-2 design must turn this green before merging.
- **NEW:** `test_prefill_state.cpp` (was started in prior attempt;
  untracked) — byte-compare recurrent state after N prefill tokens between
  `n_spec_max=0` and `n_spec_max>0`. Must be byte-identical. This catches
  the prior attempt's `n_parallel>1` corruption.

Phase 0 gate: spiral and prefill-state tests green on master (they should
be, since master doesn't widen). Trajectory test failing is the open
problem to solve before phase 2.

### Phase 1 — memory layer slot widening (no kernel changes yet)

- `src/llama-memory-recurrent.{h,cpp}`:
  - Constructor takes `n_spec_max` (additional axis size).
  - Allocate `r_l[il]`, `s_l[il]` with extra rows for slot 1..n_spec_max.
  - Add `active_slots[seq]` vector, `set_active_slot` / `get_active_slot`.
  - Modify `llama_memory_recurrent_context::s_copy(i)` to add the active
    slot's row offset, with one-shot consume reset to 0.
  - Partial `seq_rm` (mid-sequence truncation) now succeeds when n_spec_max > 0.
- `src/llama-memory-hybrid.{h,cpp}`: thread `n_spec_max` through to mem_recr;
  expose `set_active_slot`.
- `src/llama-context.{h,cpp}` + `include/llama.h`: add
  `llama_context_params::n_spec_max`, `llama_memory_set_active_slot` C API.
- `common/common.cpp` (`common_context_params_to_llama`): set `n_spec_max`
  from `params.speculative.n_max` when `--spec-type` is given.
- `src/llama-graph.cpp::build_rs`: reshape over the widened rows; guard the
  copy path for the empty range.

**Gates before merging phase 1:**
- `test_prefill_state` byte-identical for `n_spec_max=0` vs `n_spec_max=2`,
  under both `-np 1` and `-np 4`.
- Spiral regression PASS with `n_spec_max=2` and `--spec-type` not set
  (i.e., widened tensor, no spec, no emit, no slot reads). Output must be
  bit-identical to non-widened.
- No throughput regression on plain T=1 generation (within FP noise).

Phase 1 is purely structural. If gates fail here, the rest of the plan is
moot until find_slot/build_rs are debugged.

### Phase 2 — kernel emit (per Section 4 decision)

Implement the chosen kernel direction (A, B, or C-driven). For each:

- New ggml op or modification: `ggml_gated_delta_net` gains an emit-mode
  output tensor `s_intermediates` of shape `[state_dim, n_tokens]` per call.
  The tensor lives in graph scratch and is `ggml_cpy`'d into `r_l`/`s_l`
  slot rows by the model graph builder.
- Backend kernels: CUDA + CPU at minimum. Metal/HIP tracking issue.
- `src/models/qwen35.cpp::build_layer_attn_linear`: when emit is needed
  (n_spec_max > 0, T > 1, T <= 1+n_spec_max), use the emit op and route
  per-token output states into `recurrent.r_l`/`s_l` slots 1..T-1.

**Gates before merging phase 2:**
- `test_gdn_emit` per-slot bit-identity PASS.
- `test_gdn_trajectory` PASS (this is the gate the prior attempt didn't
  have and didn't notice was failing).
- Spiral regression PASS with `--spec-type mtp` and the slot reads triggered
  via partial-accept simulation (write a probe that forces active_slot != 0).
- Spiral regression PASS at n_predict=500 AND n_predict=1000 — the prior
  attempt was coherent at 400 but spiraled at 500. Must clear both.

### Phase 3 — server / spec verify integration

- `common/speculative.cpp::common_speculative_state_mtp::accept(M)`:
  set `active_slot = M+1` on the seq, trim `mtp_n_hidden` to `M+1`.
- `tools/server/server-context.cpp`: on partial accept, just call
  `llama_memory_set_active_slot(mem, slot.id, M+1)` — no ckpt save, no
  ckpt restore, no reverify, no snapshot. The next decode reads slot[M+1]
  and proceeds as if from a freshly-committed state.
- Counter fix carried forward: update `n_draft_accepted`/`n_draft_total`
  on every iteration including partial-accept. Uncovers true reject rate.

**Gates before merging phase 3:**
- Spiral regression PASS at K=1, K=2, K=3 on Roman 600.
- Acceptance counter honest (matches per-iteration debug trace).
- Throughput at K=2 ≥ 1.5× baseline (Stage 2 target from earlier plan
  iteration; confirms we recouped the work).
- Spec verify uses *no* ckpt path for hybrid. Server log has zero "restoring
  speculative checkpoint" lines on rejects.

### Phase 4 — MTP-specific work (deferred)

Once phases 1-3 prove the slot infrastructure on hybrid memory generally
(ideally validated with a separate small draft model so reject rate is
not artificially zero), then re-engage MTP:

- Verify the chained-MTP draft gen (`common_speculative_state_mtp::draft`)
  doesn't introduce its own state pollution into the slot system.
- Investigate whether `mtp_h_cache` needs parallel slot widening.
- Consider: at 100% acceptance (which MTP greedy currently hits — see
  session notes), the slot path doesn't fire much. Its value is for
  divergent draft setups (smaller models, temp>0 with proper rejection
  sampling). MTP may benefit only marginally.

---

## 6. Out of scope

- **Snapshot/restore on the existing single-slot tensor.** Was prototyped this
  session, ~9-15% throughput win at 100% MTP accept (where restore never
  fires). Not worth maintaining as a parallel path once slots land.
- **`spec_ckpt`-side reverify-elimination.** Was prototyped this session —
  saves one re-decode per reject but needs careful prompt-token /
  sampler / streaming bookkeeping that duplicates the full-accept path.
  Slots make it obsolete.
- **Mamba SSM-scan support.** `ggml_ssm_scan` would need the same emit-
  intermediates treatment. Tracking issue, not blocking GDN.
- **EAGLE3 or D>1 MTP.** Orthogonal to the memory layer.
- **Counter honesty fix in `sampling.cpp`-adjacent code.** Worth landing
  independently as a small PR; the trivial-reverify-only counter masks
  every spec setup's real reject rate, not just hybrid.

---

## 7. Files that will be touched (estimate)

Phase 0: ~250 LOC of test code (mostly salvaging existing untracked tests).

Phase 1: ~300 LOC.
- `src/llama-memory.h`
- `src/llama-memory-recurrent.{h,cpp}`
- `src/llama-memory-hybrid.{h,cpp}`
- `src/llama-context.{h,cpp}`
- `src/llama-graph.cpp::build_rs`
- `include/llama.h` (`llama_memory_set_active_slot`)
- `common/common.cpp`

Phase 2: ~600 LOC (most of this is the GDN kernel emit path on CUDA + CPU,
plus wiring in qwen35.cpp).
- `ggml/include/ggml.h`, `ggml/src/ggml.c` (op declaration)
- `ggml/src/ggml-cuda/gated_delta_net.cu` (EMIT template)
- `ggml/src/ggml-cpu/ops.cpp` (CPU reference)
- `src/models/qwen35.cpp::build_layer_attn_linear`
- `src/models/delta-net-base.cpp`

Phase 3: ~100 LOC.
- `common/speculative.cpp::common_speculative_state_mtp::accept`
- `tools/server/server-context.cpp` (replace ckpt branch with slot select)

Phase 4: TBD.

---

## 8. Where to start

1. Salvage `scripts/test_gdn_emit.cpp`, `scripts/test_prefill_state.cpp`,
   `scripts/test_slot_recovery.cpp`, `scripts/test_layer_diff.cpp`,
   `scripts/test_rollback_repro.cpp` from the working tree (untracked from
   the prior attempt). These are the diagnostic kit. Decide which to keep,
   port to `tests/` proper, register in CMakeLists.
2. Write `test_gdn_trajectory` (Section 5 phase 0). Run against master.
   Confirm it fails at ~19.65 on Qwen3.6.
3. Spend the timebox on option C from Section 4. If unresolved, lock in
   option B and proceed to Phase 1.
4. Phase 1 in isolation, with its three gates green, before any kernel work.

The earlier session's working tree has untracked files that started this
investigation. Do not delete them blindly — they encode debugging state
that took real effort to produce.
