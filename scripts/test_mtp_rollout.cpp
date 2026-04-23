// Iterated MTP sanity test. Greedy-decodes N tokens from a prompt with MTP
// drafting enabled. At every step logs:
//    main argmax  (== the token we commit next — this is t_{+1})
//    mtp  argmax  (== what our MTP head says is t_{+2})
// Then reports two agreement rates over the rollout:
//    t+1 rate : mtp[i] == main[i]      ← high means MTP predicts same position as main
//    t+2 rate : mtp[i] == main[i+1]    ← high means MTP correctly looks one further
//
// If t+1 >> t+2 our fused dispatch is off-by-one (MTP is just restating main's
// prediction, useless for speculative decoding). If t+2 is meaningful (>60% on
// a coherent prompt), MTP is doing what it's supposed to and server integration
// will give real speedup.

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static int argmax(const float * p, int32_t n) {
    int best = 0;
    float bv = p[0];
    for (int i = 1; i < n; ++i) if (p[i] > bv) { bv = p[i]; best = i; }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf> [prompt] [n_steps]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string prompt = (argc >= 3) ? argv[2] :
        "The history of the Roman Empire is one of the most complex and dramatic narratives "
        "in human civilization. Starting from its legendary founding by Romulus in 753 BCE, "
        "Rome grew from a small settlement on the Italian peninsula into an empire that "
        "eventually controlled most of the known world. The Republic, established around "
        "509 BCE, governed through a complex system of ";
    const int n_steps = (argc >= 4) ? std::stoi(argv[3]) : 60;

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const int32_t n_mtp = llama_model_n_mtp(model);
    fprintf(stderr, "llama_model_n_mtp = %d\n", n_mtp);
    if (n_mtp == 0) { fprintf(stderr, "model has no MTP head\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 4096;
    llama_context * ctx = llama_init_from_model(model, cparams);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), true, false);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), toks.data(), (int32_t)toks.size(), true, false); }
    toks.resize(n);
    fprintf(stderr, "prompt: %d tokens, rollout: %d steps\n\n", n, n_steps);

    auto piece = [&](llama_token t) -> std::string {
        std::vector<char> buf(64);
        int l = llama_token_to_piece(vocab, t, buf.data(), (int32_t)buf.size(), 0, true);
        return std::string(buf.data(), l > 0 ? l : 0);
    };

    // --- Prefill with MTP off ---
    llama_set_mtp_drafting(ctx, false);
    {
        llama_batch b = llama_batch_init((int32_t)toks.size(), 0, 1);
        b.n_tokens = (int32_t)toks.size();
        for (int i = 0; i < b.n_tokens; ++i) {
            b.token[i]    = toks[i];
            b.pos[i]      = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0]= 0;
            b.logits[i]   = (i == b.n_tokens - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
        llama_batch_free(b);
    }

    llama_token sampled = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
    llama_pos   pos     = n;      // next decode position

    // --- Rollout with MTP on ---
    llama_set_mtp_drafting(ctx, true);

    std::vector<llama_token> main_at_step(n_steps);
    std::vector<llama_token> mtp_at_step (n_steps);

    for (int s = 0; s < n_steps; ++s) {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens     = 1;
        b.token[0]     = sampled;
        b.pos[0]       = pos;
        b.n_seq_id[0]  = 1;
        b.seq_id[0][0] = 0;
        b.logits[0]    = 1;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at step %d\n", s); return 1; }
        llama_batch_free(b);

        float * mlogits = llama_get_logits_ith(ctx, -1);
        float * tlogits = llama_get_mtp_logits_ith(ctx, -1, 0);
        if (!mlogits || !tlogits) { fprintf(stderr, "null logits at step %d\n", s); return 1; }

        const int main_next = argmax(mlogits, n_vocab);
        const int mtp_draft = argmax(tlogits, n_vocab);

        main_at_step[s] = main_next;
        mtp_at_step[s]  = mtp_draft;

        sampled = main_next;
        pos    += 1;
    }

    // --- Report ---
    fprintf(stderr, "%4s  %-24s  %-24s\n", "step", "main -> t+1", "mtp  -> t+2 (claimed)");
    fprintf(stderr, "----  ------------------------  ------------------------\n");
    for (int s = 0; s < n_steps; ++s) {
        std::string m = piece(main_at_step[s]);
        std::string t = piece(mtp_at_step[s]);
        if (m.size() > 22) m = m.substr(0, 22);
        if (t.size() > 22) t = t.substr(0, 22);
        fprintf(stderr, "%4d  %-24s  %-24s\n", s, m.c_str(), t.c_str());
    }

    int t1 = 0, t2 = 0;
    for (int i = 0; i < n_steps;     ++i) if (mtp_at_step[i] == main_at_step[i])     t1++;
    for (int i = 0; i < n_steps - 1; ++i) if (mtp_at_step[i] == main_at_step[i + 1]) t2++;

    fprintf(stderr, "\n=== agreement over %d steps ===\n", n_steps);
    fprintf(stderr, "t+1 rate (mtp[i] == main[i])     : %3d / %d  = %5.1f%%\n",
            t1, n_steps,     100.0 * t1 / n_steps);
    fprintf(stderr, "t+2 rate (mtp[i] == main[i+1])   : %3d / %d  = %5.1f%%\n",
            t2, n_steps - 1, 100.0 * t2 / (n_steps - 1));
    fprintf(stderr, "\n");
    if (t1 > t2 + n_steps / 10) {
        fprintf(stderr, "VERDICT: t+1 >> t+2.  MTP is restating main's prediction. Fused dispatch is off-by-one; MTP needs to be conditioned on the just-sampled token, not the batch input.\n");
    } else if (t2 > 0.3 * (n_steps - 1)) {
        fprintf(stderr, "VERDICT: t+2 rate is substantial. MTP is producing real lookahead drafts.\n");
    } else {
        fprintf(stderr, "VERDICT: inconclusive — neither t+1 nor t+2 dominates.\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
