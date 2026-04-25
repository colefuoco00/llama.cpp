// Focused unit test for ggml_gated_delta_net_emit (CPU).
//
// Checks two invariants:
//   1. emit's last snapshot (emit[T-1]) == non-emit's final state, byte-exact.
//   2. emit's attn_scores section == non-emit's attn_scores, byte-exact.
//
// Both ops are issued in the SAME graph against the SAME random inputs so
// any divergence is purely algorithmic (no FP-noise excuse).
//
// Doesn't yet check emit[k < T-1] (intermediate snapshots) — that requires
// running a separate non-emit graph with k+1 tokens and comparing finals.
// TODO: add that as a follow-up; the all-tokens final check is the main
// regression gate for "did I break the math?"
//
// Usage: ./build/bin/test-gdn-emit

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void fill_random(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    const size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
    ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(float));
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "cpu backend init failed\n"); return 2; }

    // problem dims
    const int64_t H        = 4;     // heads
    const int64_t S_v      = 16;    // head dim
    const int64_t T        = 4;     // tokens
    const int64_t n_seqs   = 1;

    ggml_init_params ip = {
        /* mem_size   */ 16 * 1024 * 1024,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, T, n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, n_seqs);
    ggml_tensor * beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, n_seqs);
    ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S_v * S_v * H, n_seqs);
    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);
    ggml_set_input(g);
    ggml_set_input(beta);
    ggml_set_input(state);

    ggml_tensor * out_plain = ggml_gated_delta_net     (ctx, q, k, v, g, beta, state);
    ggml_tensor * out_emit  = ggml_gated_delta_net_emit(ctx, q, k, v, g, beta, state);
    ggml_set_output(out_plain);
    ggml_set_output(out_emit);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_plain);
    ggml_build_forward_expand(gf, out_emit);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }

    std::mt19937 rng(0x1234);
    fill_random(q,     rng);
    fill_random(k,     rng);
    fill_random(v,     rng);
    fill_random(g,     rng);
    fill_random(beta,  rng);
    fill_random(state, rng);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "gallocr_alloc_graph failed\n"); return 2; }
    ggml_backend_graph_compute(backend, gf);

    // Read outputs.
    const int64_t attn_elems  = S_v * H * T * n_seqs;
    const int64_t state_elems = S_v * S_v * H * n_seqs;

    std::vector<float> plain_buf(ggml_nelements(out_plain));
    ggml_backend_tensor_get(out_plain, plain_buf.data(), 0, plain_buf.size() * sizeof(float));
    std::vector<float> emit_buf(ggml_nelements(out_emit));
    ggml_backend_tensor_get(out_emit, emit_buf.data(), 0, emit_buf.size() * sizeof(float));

    // 1. attn_scores must match.
    double attn_mad = 0.0;
    for (int64_t i = 0; i < attn_elems; ++i) {
        attn_mad = std::max(attn_mad, (double) std::fabs(plain_buf[i] - emit_buf[i]));
    }
    fprintf(stderr, "attn_scores max_abs_diff = %g\n", attn_mad);

    // 2. emit[T-1] (last snapshot) must equal non-emit's final state.
    const float * plain_state    = plain_buf.data() + attn_elems;
    const float * emit_state_last = emit_buf.data()  + attn_elems + (T - 1) * state_elems;
    double state_mad = 0.0;
    for (int64_t i = 0; i < state_elems; ++i) {
        state_mad = std::max(state_mad, (double) std::fabs(plain_state[i] - emit_state_last[i]));
    }
    fprintf(stderr, "emit[T-1] vs non-emit final  max_abs_diff = %g\n", state_mad);

    const double tol = 1e-6;  // CPU fp32, same kernel path, must be exact
    int rc = (attn_mad <= tol && state_mad <= tol) ? 0 : 1;
    fprintf(stderr, "%s\n", rc == 0 ? "PASS" : "FAIL");

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return rc;
}
