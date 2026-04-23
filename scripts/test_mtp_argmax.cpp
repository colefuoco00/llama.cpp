// Sanity test for the MTP draft head on a Qwen3.5/3.6 GGUF that preserved its
// NextN tensors. Runs:
//   - prefill on a fixed prompt (MTP should NOT dispatch: n_tokens > n_outputs)
//   - one decode step with llama_set_mtp_drafting(true) (MTP dispatches)
//   - prints argmax(main_logits) and argmax(mtp_logits) for the last output
//
// Expected behavior:
//   * llama_model_n_mtp(model) == 1 on a Qwen3.5/3.6 MTP GGUF
//   * llama_get_mtp_logits_ith returns non-null after the decode step
//   * argmax(main) and argmax(mtp) differ: main predicts t+1, MTP predicts t+2
//
// Build (after cmake configured):
//   g++ -std=c++17 scripts/test_mtp_argmax.cpp \
//       -I include -I common \
//       -L build/bin -lllama -lllama-common -lggml -lggml-base \
//       -Wl,-rpath,build/bin -o build/bin/test_mtp_argmax

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int argmax(const float * p, int32_t n) {
    int best = 0;
    float bv = p[0];
    for (int i = 1; i < n; ++i) {
        if (p[i] > bv) { bv = p[i]; best = i; }
    }
    return best;
}

static void top_k(const float * p, int32_t n, int k, std::vector<std::pair<int,float>> & out) {
    out.clear();
    out.reserve(k);
    for (int i = 0; i < n; ++i) {
        if ((int)out.size() < k) {
            out.emplace_back(i, p[i]);
            std::push_heap(out.begin(), out.end(), [](auto&a, auto&b){return a.second > b.second;});
        } else if (p[i] > out.front().second) {
            std::pop_heap(out.begin(), out.end(), [](auto&a, auto&b){return a.second > b.second;});
            out.back() = {i, p[i]};
            std::push_heap(out.begin(), out.end(), [](auto&a, auto&b){return a.second > b.second;});
        }
    }
    std::sort(out.begin(), out.end(), [](auto&a, auto&b){return a.second > b.second;});
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <gguf> [prompt]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string prompt = (argc >= 3) ? argv[2] : "The capital of France is";

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const int32_t n_mtp = llama_model_n_mtp(model);
    printf("llama_model_n_mtp = %d\n", n_mtp);
    if (n_mtp == 0) { fprintf(stderr, "model has no MTP head\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // tokenize
    std::vector<llama_token> toks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                           toks.data(), (int32_t)toks.size(), true, false);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), toks.data(), (int32_t)toks.size(), true, false); }
    toks.resize(n);

    printf("prompt tokens: %d\n", n);

    // Step A: prefill with MTP off. Last-token logits only.
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

        float * mtp = llama_get_mtp_logits_ith(ctx, -1, 0);
        printf("prefill mtp ptr (flag off): %p  (expected null)\n", (void*)mtp);
        llama_batch_free(b);
    }

    // Step B: enable MTP, decode ONE more token. All logits requested.
    llama_set_mtp_drafting(ctx, true);

    // Seed the single-token decode with the argmax of the prefill's main logits.
    float * prefill_main = llama_get_logits_ith(ctx, -1);
    const int sampled = argmax(prefill_main, n_vocab);

    std::vector<char> piece(128);
    int np = llama_token_to_piece(vocab, sampled, piece.data(), (int32_t)piece.size(), 0, true);
    printf("prefill argmax token: %d (\"%.*s\")\n", sampled, np > 0 ? np : 0, piece.data());

    {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens     = 1;
        b.token[0]     = sampled;
        b.pos[0]       = n;
        b.n_seq_id[0]  = 1;
        b.seq_id[0][0] = 0;
        b.logits[0]    = 1;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

        float * main_logits = llama_get_logits_ith(ctx, -1);
        float * mtp_logits  = llama_get_mtp_logits_ith(ctx, -1, 0);

        if (!main_logits) { fprintf(stderr, "no main logits\n"); return 1; }
        if (!mtp_logits)  { fprintf(stderr, "MTP logits null — MTP did not dispatch on decode\n"); return 1; }

        const int main_top = argmax(main_logits, n_vocab);
        const int mtp_top  = argmax(mtp_logits,  n_vocab);

        std::vector<char> main_piece(128), mtp_piece(128);
        int mlen = llama_token_to_piece(vocab, main_top, main_piece.data(), (int32_t)main_piece.size(), 0, true);
        int tlen = llama_token_to_piece(vocab, mtp_top,  mtp_piece.data(),  (int32_t)mtp_piece.size(),  0, true);

        printf("\n--- after 1 decode step ---\n");
        printf("  main argmax : %d \"%.*s\"   (predicts t+1)\n", main_top, mlen > 0 ? mlen : 0, main_piece.data());
        printf("  mtp  argmax : %d \"%.*s\"   (predicts t+2)\n", mtp_top,  tlen > 0 ? tlen : 0, mtp_piece.data());

        printf("\nmain top-5:\n");
        std::vector<std::pair<int,float>> topk;
        top_k(main_logits, n_vocab, 5, topk);
        for (auto & [id, sc] : topk) {
            std::vector<char> buf(64);
            int l = llama_token_to_piece(vocab, id, buf.data(), (int32_t)buf.size(), 0, true);
            printf("  %6d  %9.3f  \"%.*s\"\n", id, sc, l > 0 ? l : 0, buf.data());
        }
        printf("\nmtp top-5:\n");
        top_k(mtp_logits, n_vocab, 5, topk);
        for (auto & [id, sc] : topk) {
            std::vector<char> buf(64);
            int l = llama_token_to_piece(vocab, id, buf.data(), (int32_t)buf.size(), 0, true);
            printf("  %6d  %9.3f  \"%.*s\"\n", id, sc, l > 0 ? l : 0, buf.data());
        }

        llama_batch_free(b);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
