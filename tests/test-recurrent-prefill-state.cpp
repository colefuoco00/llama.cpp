// Phase 1 (memory slot widening) byte-identity gate.
//
// Verifies that adding (1+n_spec_max) state slots to the recurrent memory
// tensors does NOT perturb the prefill state when no slot reads/writes are
// triggered. Slot 0 must hold the same bytes as the non-widened tensor's only
// slot, since slot 0 IS the active state during normal decode.
//
// Direct memory comparison isn't possible without leaking internals, so we
// compare end-to-end: post-prefill+decode logits must be bit-identical
// (within tight FP) between n_spec_max=0 and n_spec_max=3.
//
// On master this would not compile (no n_spec_max field) — runs only after
// llama_context_params::n_spec_max is added (Phase 1).
//
// Usage:
//   ./build/bin/test-recurrent-prefill-state <model.gguf>

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char * PROMPT =
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs.";

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

static int decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.token   [0] = tok;
    b.pos     [0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id  [0][0] = 0;
    b.logits  [0] = 1;
    b.n_tokens = 1;
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static std::vector<float> capture_logits_at(llama_context * ctx, const llama_model * model, int32_t i) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * src = llama_get_logits_ith(ctx, i);
    if (!src) return {};
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

static std::vector<float> run(llama_model * model, const std::vector<llama_token> & prompt,
                              uint32_t n_spec_max) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx       = 4096;
    cp.n_batch     = 512;
    cp.n_ubatch    = 512;
    cp.no_perf     = true;
    cp.n_spec_max  = n_spec_max;

    llama_context * c = llama_init_from_model(model, cp);
    if (!c) { fprintf(stderr, "ctx build failed (n_spec_max=%u)\n", n_spec_max); return {}; }

    if (decode_batch(c, prompt, 0, /*last logits only*/ true) != 0) {
        fprintf(stderr, "prefill failed (n_spec_max=%u)\n", n_spec_max);
        llama_free(c);
        return {};
    }
    if (decode_one(c, /*probe*/ 0, (llama_pos) prompt.size()) != 0) {
        fprintf(stderr, "probe decode failed (n_spec_max=%u)\n", n_spec_max);
        llama_free(c);
        return {};
    }
    llama_synchronize(c);

    auto logits = capture_logits_at(c, model, 0);
    llama_free(c);
    return logits;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_prompt = -llama_tokenize(vocab, PROMPT, (int) strlen(PROMPT), nullptr, 0, true, false);
    std::vector<llama_token> prompt(n_prompt);
    if (llama_tokenize(vocab, PROMPT, (int) strlen(PROMPT), prompt.data(), n_prompt, true, false) < 0) {
        fprintf(stderr, "tokenize failed\n"); return 2;
    }
    fprintf(stderr, "prompt: %d tokens\n", n_prompt);

    fprintf(stderr, "running n_spec_max=0 ...\n");
    auto logits_0 = run(model, prompt, 0);
    if (logits_0.empty()) { llama_model_free(model); llama_backend_free(); return 2; }

    fprintf(stderr, "running n_spec_max=3 ...\n");
    auto logits_3 = run(model, prompt, 3);
    if (logits_3.empty()) { llama_model_free(model); llama_backend_free(); return 2; }

    const double mad = max_abs_diff(logits_0, logits_3);
    fprintf(stderr,
            "max_abs_diff(n_spec_max=0, n_spec_max=3) = %g  (n_vocab=%zu)\n",
            mad, logits_0.size());

    // Slot widening with no slot reads/writes must not perturb output. Tight tol.
    const double tol = 1e-4;

    int rc = (mad <= tol) ? 0 : 1;
    fprintf(stderr, "%s: max_abs_diff %g  tol %g\n", rc == 0 ? "PASS" : "FAIL", mad, tol);

    llama_model_free(model);
    llama_backend_free();
    return rc;
}
