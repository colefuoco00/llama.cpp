// Empirical test: does target's argmax at each position differ between
// sequential decode (one token per llama_decode call) and batched decode
// (all tokens in one call)?
//
// If batched and sequential agree at every position, then there's no
// "FP-batch-flip" hypothesis. If they disagree even at a few positions,
// the hypothesis is plausible.
//
// Usage:
//   ./build/bin/test-batched-vs-sequential <gguf>

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char * PROMPT =
    "In the ancient city of Rome, the Senate gathered in solemn assembly "
    "as storm clouds darkened the sky above the Capitoline Hill. Senators "
    "in their white togas debated the fate of the Republic while generals "
    "returned from distant conquests bringing news of new lands subjugated "
    "and countless treasures hauled back to the capital.";

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

static int decode_batch(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token   [b.n_tokens] = toks[i];
        b.pos     [b.n_tokens] = pos0 + (llama_pos) i;
        b.n_seq_id[b.n_tokens] = 1;
        b.seq_id  [b.n_tokens][0] = 0;
        b.logits  [b.n_tokens] = 1;  // need logits at every position
        b.n_tokens++;
    }
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static llama_token argmax_at(llama_context * ctx, const llama_model * model, int32_t i) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * src = llama_get_logits_ith(ctx, i);
    if (!src) {
        fprintf(stderr, "logits NULL at i=%d\n", i);
        return -1;
    }
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

    // We want a batch of "next-N tokens after some prefix" to compare
    // sequential vs batched. Use the model itself to generate N tokens
    // sequentially first (greedy continuation), then test sequential vs
    // batched re-decode of the SAME N tokens.

    auto build_ctx = [&]() -> llama_context * {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx     = 4096;
        cp.n_batch   = 512;
        cp.n_ubatch  = 512;
        cp.no_perf   = true;
        return llama_init_from_model(model, cp);
    };

    // 1. Generate N continuation tokens sequentially using greedy decode
    const int N = 80;
    std::vector<llama_token> cont; cont.reserve(N);

    {
        llama_context * c = build_ctx();
        if (!c) { fprintf(stderr, "ctx build failed\n"); return 2; }
        // prefill the prompt
        if (decode_batch(c, prompt, 0) != 0) { fprintf(stderr, "prefill failed\n"); return 2; }

        llama_pos pos = (llama_pos) n_prompt;
        for (int i = 0; i < N; ++i) {
            // logits for the position just decoded predict the NEXT token
            llama_token next = argmax_at(c, model, /*last logits idx*/ -1);  // -1 = last
            // llama_get_logits_ith with -1 doesn't work; capture from logits at 0
            // since the last batch has only 1 logit
            // (decode_batch sets logits[i]=1 for every token; for batch of size 1,
            // index 0 is the only logits row)
            cont.push_back(next);
            if (decode_one(c, next, pos, true) != 0) { fprintf(stderr, "seq decode failed\n"); return 2; }
            pos++;
        }
        llama_free(c);
    }

    fprintf(stderr, "generated %zu continuation tokens\n", cont.size());

    // 2. Sequential re-decode: for each position p in [n_prompt, n_prompt+N),
    //    capture argmax (= what target predicts AT that position from prefix).
    std::vector<llama_token> argmax_seq(N);
    {
        llama_context * c = build_ctx();
        if (decode_batch(c, prompt, 0) != 0) { fprintf(stderr, "prefill seq failed\n"); return 2; }

        for (int i = 0; i < N; ++i) {
            // After prefill, decode cont[0..i-1] sequentially, then capture argmax
            // at the LAST position which predicts cont[i].
            // Actually: we want target's argmax at position (n_prompt + i - 1)
            // which predicts cont[i]. After feeding cont[i-1] at pos n_prompt+i-1,
            // logits are at that position predicting next. So we should capture
            // BEFORE feeding cont[i].
            //
            // Simpler: run from prefill, after each cont[j] decode capture argmax.
            // The argmax after decoding cont[j] predicts what comes AFTER cont[j],
            // i.e., cont[j+1].
            // So argmax_seq[j+1] = argmax after decoding cont[j].
            // For j=0 (predicting cont[0]) we use argmax after prefill.
            (void)i;
        }

        // Easier: capture argmax after prefill (predicts cont[0]), then loop.
        argmax_seq[0] = argmax_at(c, model, 0);  // batch size was n_prompt; last index = n_prompt-1
        // Wait, decode_batch set logits at every position. Need logits at index n_prompt-1.

        // Re-do prefill with only last-position logits for simplicity.
        llama_free(c);
        c = build_ctx();
        // prefill but only request logits at the last position
        {
            llama_batch b = llama_batch_init(n_prompt, 0, 1);
            for (int i = 0; i < n_prompt; ++i) {
                b.token[b.n_tokens] = prompt[i];
                b.pos[b.n_tokens] = i;
                b.n_seq_id[b.n_tokens] = 1;
                b.seq_id[b.n_tokens][0] = 0;
                b.logits[b.n_tokens] = (i == n_prompt - 1) ? 1 : 0;
                b.n_tokens++;
            }
            if (llama_decode(c, b) != 0) { fprintf(stderr, "prefill failed\n"); return 2; }
            llama_batch_free(b);
        }
        argmax_seq[0] = argmax_at(c, model, 0);

        // Then for each i in 1..N-1: feed cont[i-1] sequentially, capture argmax.
        for (int i = 1; i < N; ++i) {
            if (decode_one(c, cont[i-1], (llama_pos)(n_prompt + i - 1), true) != 0) {
                fprintf(stderr, "seq decode i=%d failed\n", i); return 2;
            }
            argmax_seq[i] = argmax_at(c, model, 0);
        }
        llama_free(c);
    }

    // 3. Batched re-decode: feed all N tokens as one batch after prefill,
    //    capture argmax at each position.
    std::vector<llama_token> argmax_bat(N);
    {
        llama_context * c = build_ctx();
        // prefill with last-only logits
        {
            llama_batch b = llama_batch_init(n_prompt, 0, 1);
            for (int i = 0; i < n_prompt; ++i) {
                b.token[b.n_tokens] = prompt[i];
                b.pos[b.n_tokens] = i;
                b.n_seq_id[b.n_tokens] = 1;
                b.seq_id[b.n_tokens][0] = 0;
                b.logits[b.n_tokens] = 0;
                b.n_tokens++;
            }
            if (llama_decode(c, b) != 0) { fprintf(stderr, "prefill batched failed\n"); return 2; }
            llama_batch_free(b);
        }

        // Now feed N continuation tokens as one batch, ask for logits everywhere.
        // We need argmax at each position predicting the NEXT token, i.e.,
        // logits at position (n_prompt-1, n_prompt, ..., n_prompt+N-2) predict
        // cont[0], cont[1], ..., cont[N-1].
        // The first prediction (cont[0]) was already captured above as argmax_seq[0]
        // -- it is the same in batched mode since prefill is the same.
        // Wait, it IS the same prefill, so argmax at n_prompt-1 from prefill = argmax_seq[0].
        // For subsequent argmax_bat[1..N-1], we feed cont[0..N-2] as a batch and
        // capture argmax at each.
        argmax_bat[0] = argmax_seq[0];  // same prefill, no batched diff possible

        std::vector<llama_token> feed(cont.begin(), cont.end() - 1);  // cont[0..N-2]
        if (!feed.empty()) {
            llama_batch b = llama_batch_init((int32_t) feed.size(), 0, 1);
            for (size_t i = 0; i < feed.size(); ++i) {
                b.token[b.n_tokens] = feed[i];
                b.pos[b.n_tokens] = (llama_pos) (n_prompt + i);
                b.n_seq_id[b.n_tokens] = 1;
                b.seq_id[b.n_tokens][0] = 0;
                b.logits[b.n_tokens] = 1;
                b.n_tokens++;
            }
            if (llama_decode(c, b) != 0) { fprintf(stderr, "batched decode failed\n"); return 2; }
            for (size_t i = 0; i < feed.size(); ++i) {
                argmax_bat[i + 1] = argmax_at(c, model, (int32_t) i);
            }
            llama_batch_free(b);
        }
        llama_free(c);
    }

    // Compare
    int disagreements = 0;
    for (int i = 0; i < N; ++i) {
        if (argmax_seq[i] != argmax_bat[i]) {
            disagreements++;
            if (disagreements <= 10) {
                fprintf(stderr, "  pos %3d: seq=%6d  bat=%6d  cont=%6d\n",
                        i, argmax_seq[i], argmax_bat[i], cont[i]);
            }
        }
    }
    fprintf(stderr, "TOTAL disagreements: %d / %d positions\n", disagreements, N);

    if (disagreements == 0) {
        fprintf(stderr, "RESULT: batched == sequential at every position. FP-batch-flip hypothesis disproved.\n");
    } else {
        fprintf(stderr, "RESULT: batched != sequential at %d/%d positions. FP-batch-flip is real.\n",
                disagreements, N);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
