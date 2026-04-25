// Recurrent partial-rollback gate test.
//
// On master, llama_memory_recurrent::seq_rm rejects any partial truncation
// that includes the cell's final position. This test verifies the END STATE
// once that limitation is removed via per-token state slots emitted by the
// recurrent kernels (gated_delta_net + ssm_conv) during a multi-token decode.
//
// Test flow:
//   ref ctx:  prefill N, decode tA tB sequentially, decode tX, capture logits.
//   test ctx: prefill N, decode [tA, tB, tC, tD] in one batch,
//             seq_rm(seq=0, p0=N+2, -1)  ←  rolls back tC, tD,
//             decode tX, capture logits.
//   assert max_abs_diff(logits_ref, logits_test) < tol.
//
// On master this test fails at the seq_rm call (returns false). After the
// slot infrastructure + GDN emit kernel land, seq_rm succeeds, the recurrent
// state is selected from slot[2] (state after 2 verify tokens), and the
// downstream decode of tX must produce logits matching the sequential
// reference within FP tolerance.
//
// Usage:
//   ./build/bin/test-recurrent-rollback <model.gguf>
//
// Exit codes: 0 = pass, 1 = correctness fail, 2 = setup error.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char * PROMPT =
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump.";

// Number of tokens to verify in the batched decode (so test rolls back T/2).
static constexpr int T_VERIFY = 4;

// Number of those tokens to keep after rollback.
static constexpr int T_KEEP = 2;

// Number of probe tokens decoded after the rollback (must be > 1 so the
// forward continuation also goes through the chunking kernel — a T=1 probe
// would land on Fused-AR which is a numerically distinct trajectory per
// findings.md, contaminating the comparison).
static constexpr int T_PROBE_BATCH = 2;

static int decode_one(llama_context * ctx, llama_token tok, llama_pos pos, bool want_logits) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.token   [0] = tok;
    b.pos     [0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id  [0][0] = 0;
    b.logits  [0] = want_logits ? 1 : 0;
    b.n_tokens = 1;
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static int decode_batch(llama_context * ctx, const std::vector<llama_token> & toks,
                        llama_pos pos0, bool logits_at_last_only) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token   [b.n_tokens] = toks[i];
        b.pos     [b.n_tokens] = pos0 + (llama_pos) i;
        b.n_seq_id[b.n_tokens] = 1;
        b.seq_id  [b.n_tokens][0] = 0;
        b.logits  [b.n_tokens] = logits_at_last_only ? (i + 1 == toks.size() ? 1 : 0) : 1;
        b.n_tokens++;
    }
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static std::vector<float> capture_last_logits(llama_context * ctx, const llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * src = llama_get_logits_ith(ctx, /*last logits*/ -1);
    if (!src) {
        // some backends return null when called with -1 in batch-of-1; fall back to idx 0
        src = llama_get_logits_ith(ctx, 0);
    }
    if (!src) {
        fprintf(stderr, "ERROR: llama_get_logits_ith returned null\n");
        return {};
    }
    return std::vector<float>(src, src + n_vocab);
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    }
    return m;
}

static llama_token argmax_at(llama_context * ctx, const llama_model * model, int32_t i) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * src = llama_get_logits_ith(ctx, i);
    if (!src) return -1;
    int best = 0;
    float bv = src[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (src[v] > bv) { bv = src[v]; best = v; }
    }
    return (llama_token) best;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    // Phase 2 emit kernel is CPU-only this iteration; force CPU offload so
    // the GDN op picks up the emit flag in op_params instead of falling
    // through to a CUDA path that ignores it.
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }

    // ref ctx: no spec slots — drives the sequential-decode reference path
    // through the existing single-slot kernel.
    // test ctx: n_spec_max sized to fit the verify batch so per-token
    // snapshots are emitted into slots 1..T-1.
    auto build_ctx = [&](uint32_t n_spec_max) -> llama_context * {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx      = 4096;
        cp.n_batch    = 512;
        cp.n_ubatch   = 512;       // need >= T_VERIFY so the verify batch isn't split into ubatches
        cp.no_perf    = true;
        cp.n_spec_max = n_spec_max;
        return llama_init_from_model(model, cp);
    };

    const llama_vocab * vocab = llama_model_get_vocab(model);

    int n_prompt = -llama_tokenize(vocab, PROMPT, (int) strlen(PROMPT), nullptr, 0, true, false);
    std::vector<llama_token> prompt(n_prompt);
    if (llama_tokenize(vocab, PROMPT, (int) strlen(PROMPT), prompt.data(), n_prompt, true, false) < 0) {
        fprintf(stderr, "tokenize failed\n"); return 2;
    }
    fprintf(stderr, "prompt: %d tokens\n", n_prompt);

    // Generate T_VERIFY continuation tokens via greedy decode in a temporary
    // context. Using the model's own continuation keeps the test deterministic
    // and uses tokens that are in-distribution for whatever model is loaded.
    std::vector<llama_token> verify_toks;
    verify_toks.reserve(T_VERIFY);
    {
        llama_context * c = build_ctx(0);
        if (!c) { fprintf(stderr, "tmp ctx build failed\n"); return 2; }
        if (decode_batch(c, prompt, 0, /*last logits only*/ true) != 0) { fprintf(stderr, "tmp prefill failed\n"); return 2; }
        llama_pos pos = (llama_pos) n_prompt;
        // logits index after prefill = n_prompt - 1 (the last position with logits=true);
        // after each subsequent single-token decode, logits index = 0.
        int32_t logits_idx = n_prompt - 1;
        for (int i = 0; i < T_VERIFY; ++i) {
            llama_token next = argmax_at(c, model, logits_idx);
            if (next < 0) { fprintf(stderr, "argmax_at failed (i=%d, idx=%d)\n", i, logits_idx); return 2; }
            verify_toks.push_back(next);
            if (decode_one(c, next, pos++, true) != 0) { fprintf(stderr, "tmp decode failed\n"); return 2; }
            logits_idx = 0;
        }
        llama_free(c);
    }

    fprintf(stderr, "verify tokens: ");
    for (auto t : verify_toks) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n");

    // Probe batch: tokens decoded after the (rolled-back) state in the test
    // ctx, and after the matching short verify in the ref ctx. Same tokens
    // in both, so any state divergence at the rollback point shows up as a
    // logits divergence at the end. Uses verify_toks[T_KEEP..T_KEEP+P-1] so
    // it's deterministic and in-distribution.
    GGML_ASSERT(T_KEEP + T_PROBE_BATCH <= T_VERIFY);
    std::vector<llama_token> probe_batch(verify_toks.begin() + T_KEEP,
                                          verify_toks.begin() + T_KEEP + T_PROBE_BATCH);

    // ===== ref ctx: prefill, batched verify[0..T_KEEP-1], batched probe =====
    // Both batched decodes land on the chunking-style trajectory (T>1).
    std::vector<float> logits_ref;
    {
        llama_context * c = build_ctx(0);
        if (!c) { fprintf(stderr, "ref ctx build failed\n"); return 2; }

        if (decode_batch(c, prompt, 0, true) != 0) { fprintf(stderr, "ref prefill failed\n"); return 2; }

        std::vector<llama_token> kept(verify_toks.begin(), verify_toks.begin() + T_KEEP);
        if (decode_batch(c, kept, (llama_pos) n_prompt, /*last logits only*/ false) != 0) {
            fprintf(stderr, "ref kept-batch decode failed\n"); return 2;
        }
        if (decode_batch(c, probe_batch, (llama_pos)(n_prompt + T_KEEP), /*last logits only*/ true) != 0) {
            fprintf(stderr, "ref probe-batch decode failed\n"); return 2;
        }
        llama_synchronize(c);
        {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            const float * src = llama_get_logits_ith(c, T_PROBE_BATCH - 1);
            if (!src) { fprintf(stderr, "ref logits idx %d null\n", T_PROBE_BATCH - 1); return 2; }
            logits_ref.assign(src, src + n_vocab);
        }
        llama_free(c);
    }

    // ===== test ctx: prefill, emit-batched T_VERIFY, seq_rm partial, batched probe =====
    std::vector<float> logits_test;
    {
        // n_spec_max sized for the verify (T_VERIFY tokens => need T_VERIFY-1
        // rollback slots beyond slot 0).
        llama_context * c = build_ctx((uint32_t)(T_VERIFY - 1));
        if (!c) { fprintf(stderr, "test ctx build failed\n"); return 2; }

        if (decode_batch(c, prompt, 0, true) != 0) { fprintf(stderr, "test prefill failed\n"); return 2; }

        // emit-batched decode of T_VERIFY tokens. Per-token states land in
        // recurrent slots 1..T_VERIFY-1; final state in slot 0.
        if (decode_batch(c, verify_toks, (llama_pos) n_prompt, /*last logits only*/ false) != 0) {
            fprintf(stderr, "test batched decode failed\n"); return 2;
        }
        llama_synchronize(c);

        // partial seq_rm: keep first T_KEEP, drop the rest. Sets active_slot
        // to (T_VERIFY - T_KEEP) so the next decode reads slot[K] (state
        // after T_KEEP tokens).
        llama_memory_t mem = llama_get_memory(c);
        const llama_pos p0 = (llama_pos)(n_prompt + T_KEEP);
        const bool ok = llama_memory_seq_rm(mem, /*seq_id*/ 0, p0, -1);

        if (!ok) {
            fprintf(stderr,
                    "FAIL: llama_memory_seq_rm(seq=0, p0=%d, -1) returned false.\n"
                    "  After slot work this should succeed.\n",
                    (int) p0);
            llama_free(c);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        llama_synchronize(c);

        // batched probe from rolled-back state — same tokens as ref's probe.
        if (decode_batch(c, probe_batch, (llama_pos)(n_prompt + T_KEEP), /*last logits only*/ true) != 0) {
            fprintf(stderr, "test probe-batch decode failed\n"); return 2;
        }
        llama_synchronize(c);
        {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            const float * src = llama_get_logits_ith(c, T_PROBE_BATCH - 1);
            if (!src) { fprintf(stderr, "test logits idx %d null\n", T_PROBE_BATCH - 1); return 2; }
            logits_test.assign(src, src + n_vocab);
        }
        llama_free(c);
    }

    if (logits_ref.empty() || logits_test.empty()) {
        fprintf(stderr, "ERROR: empty logits captured\n");
        return 2;
    }

    const double mad = max_abs_diff(logits_ref, logits_test);
    fprintf(stderr,
            "max_abs_diff(ref, rolled_back) = %g  (n_vocab = %zu, T_VERIFY=%d, T_KEEP=%d)\n",
            mad, logits_ref.size(), T_VERIFY, T_KEEP);

    // Tolerance — fp32 path on CPU should be bit-exact (mad ~= 0). Allow a small
    // slack for any reduction-order differences. If this fires, slot select is
    // pulling state from the wrong place or the kernel emit is producing a
    // different trajectory than the per-token sequential decode.
    const double tol = 1e-3;

    int rc = (mad <= tol) ? 0 : 1;
    fprintf(stderr, "%s: max_abs_diff %g  tol %g\n", rc == 0 ? "PASS" : "FAIL", mad, tol);

    llama_model_free(model);
    llama_backend_free();
    return rc;
}
