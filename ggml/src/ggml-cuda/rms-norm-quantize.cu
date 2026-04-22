#include "rms-norm-quantize.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "mmid.cuh"
#include "quantize.cuh"

#include <cstdint>

// -----------------------------------------------------------------------------
// Fused RMS_NORM + MUL + quantize-to-Q8_1 (block_q8_1_mmq layout, for MMQ path)
// -----------------------------------------------------------------------------

template <int block_size, mmq_q8_1_ds_layout ds_layout, bool write_f32, bool use_ids>
__launch_bounds__(block_size, 1)
static __global__ void rms_norm_mul_quantize_mmq_q8_1_kernel(
        const float   * __restrict__ x,
        const float   * __restrict__ mul,
        const int32_t * __restrict__ ids,
        void          * __restrict__ vy,
        float         * __restrict__ dst_f32,
        const int     ncols,
        const int     ncols_padded,
        const int64_t stride_row,
        const int64_t stride_channel,
        const int64_t stride_sample,
        const int64_t mul_stride_row,
        const int64_t mul_stride_channel,
        const int64_t mul_stride_sample,
        const uint3   mul_ncols_packed,
        const uint3   mul_nrows_packed,
        const uint3   mul_nchannels_packed,
        const uint3   mul_nsamples_packed,
        const float   eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    // For MoE (use_ids): blockIdx.x indexes the flat (token, expert) route, and
    // ids[blockIdx.x] maps back to the source token row. F32 writeback also targets
    // the source row so the MUL tensor retains per-token semantics (multiple routes
    // for the same token may race-write the same location with identical values).
    const int route      = blockIdx.x;
    const int src_row    = use_ids ? ids[route] : route;
    const int channel    = blockIdx.y;
    const int sample     = blockIdx.z;
    const int tid        = threadIdx.x;

    const float * x_row = x + sample * stride_sample + channel * stride_channel + src_row * stride_row;

    const uint32_t mr = fastmodulo(src_row, mul_nrows_packed);
    const uint32_t mc = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t ms = fastmodulo(sample,  mul_nsamples_packed);
    const float * mul_row = mul + ms * mul_stride_sample + mc * mul_stride_channel + mr * mul_stride_row;

    // ---- Phase 1: sum of squares over the (unpadded) row ----
    float sumsq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x_row[col];
        sumsq += xi * xi;
    }

    extern __shared__ float s_sum[];
    sumsq = block_reduce<block_reduce_method::SUM, block_size>(sumsq, s_sum);

    const float rms_scale = rsqrtf(sumsq / ncols + eps);

    // ---- Phase 2+3: normalize, multiply by weight, quantize, write ----
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    // F32 output row base. In the MoE case the MUL output is per-token (not per-route),
    // so write to the source-token row. Multiple routes to the same token produce the
    // same F32 values, so concurrent writes to the same location are benign.
    float * dst_row = nullptr;
    if constexpr (write_f32) {
        dst_row = dst_f32 + ((int64_t)(sample * nchannels + channel) * nrows + src_row) * ncols;
    }

    // Output base block index for (sample, channel) slab, matching quantize_mmq_q8_1:
    //   ib0 = blockIdx.z * ((gridDim.x * gridDim.y * blockDim.x) / QK8_1)
    // The "row" dimension of the output is indexed by `route` (the flat output row
    // including both the MoE route and the unflattened batch row when use_ids=false).
    const int64_t batch_flat = (int64_t) sample * nchannels + channel;
    const int64_t blocks_per_slab = (int64_t) nrows * (ncols_padded / (4 * QK8_1));
    const int64_t ib0 = batch_flat * blocks_per_slab;

    // Each thread processes 4 consecutive values per iteration. A warp (32 lanes × 4 = 128 vals) fills
    // exactly one block_q8_1_mmq. The stride per iteration is block_size*4 values.
    for (int i0 = tid * 4; i0 < ncols_padded; i0 += block_size * 4) {
        float4 xi;
        if (i0 + 3 < ncols) {
            xi = *reinterpret_cast<const float4 *>(&x_row[i0]);
        } else {
            xi.x = (i0 + 0 < ncols) ? x_row[i0 + 0] : 0.0f;
            xi.y = (i0 + 1 < ncols) ? x_row[i0 + 1] : 0.0f;
            xi.z = (i0 + 2 < ncols) ? x_row[i0 + 2] : 0.0f;
            xi.w = (i0 + 3 < ncols) ? x_row[i0 + 3] : 0.0f;
        }

        // Apply RMS normalization and the per-column weight.
        const uint32_t mcol0 = fastmodulo((uint32_t)(i0 + 0), mul_ncols_packed);
        const uint32_t mcol1 = fastmodulo((uint32_t)(i0 + 1), mul_ncols_packed);
        const uint32_t mcol2 = fastmodulo((uint32_t)(i0 + 2), mul_ncols_packed);
        const uint32_t mcol3 = fastmodulo((uint32_t)(i0 + 3), mul_ncols_packed);
        xi.x = rms_scale * xi.x * mul_row[mcol0];
        xi.y = rms_scale * xi.y * mul_row[mcol1];
        xi.z = rms_scale * xi.z * mul_row[mcol2];
        xi.w = rms_scale * xi.w * mul_row[mcol3];

        if constexpr (write_f32) {
            // Write the F32 post-norm+mul values to the original MUL output tensor so that
            // any other consumers of the MUL output (e.g. parallel Q/K/V projections, embedding
            // output) still see the expected data.
            if (i0 + 3 < ncols) {
                *reinterpret_cast<float4 *>(&dst_row[i0]) = xi;
            } else {
                if (i0 + 0 < ncols) dst_row[i0 + 0] = xi.x;
                if (i0 + 1 < ncols) dst_row[i0 + 1] = xi.y;
                if (i0 + 2 < ncols) dst_row[i0 + 2] = xi.z;
                if (i0 + 3 < ncols) dst_row[i0 + 3] = xi.w;
            }
        }

        // amax reduction across vals_per_scale/4 lanes (8 lanes for scale=32, 16 lanes for scale=64)
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
        for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
        }

        float sum = 0.0f;
        if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
            sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
            for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
            }
        }

        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x * d_inv);
        q.y = roundf(xi.y * d_inv);
        q.z = roundf(xi.z * d_inv);
        q.w = roundf(xi.w * d_inv);

        const int64_t ib  = ib0 + (int64_t)(i0 / (4 * QK8_1)) * nrows + route;
        const int     iqs = i0 % (4 * QK8_1);

        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs / 4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if ((iqs % 16) == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs / 16] = sum;
            }
            if ((iqs % 64) == 0) {
                const float d = 1.0f / d_inv;
                y[ib].d2s6[iqs / 64] = d;
            }
            continue;
        }

        if ((iqs % 32) != 0) {
            continue;
        }
        const float d = 1.0f / d_inv;
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

// -----------------------------------------------------------------------------
// Fused RMS_NORM + MUL + quantize-to-Q8_1 (per-token, MoE variant)
//
// Grid: (n_tokens, 1, 1). Each block handles one source token: RMS+MUL is done once
// per token, and the resulting Q8_1 data is written to every route output position
// that sources this token (routes sharing the same token produce identical Q8_1,
// so we compute once and fan out the writes). This avoids the O(n_expert_used)
// redundant RMS work the naive "one block per route" approach would do.
//
// Routes for the token are discovered by a block-cooperative scan of ids_src1 into
// shared memory. MMID_MAX_ROUTES_PER_TOKEN caps how many routes per token we handle
// (typical n_expert_used ≤ 8, with plenty of headroom).
// -----------------------------------------------------------------------------

#define MMID_MAX_ROUTES_PER_TOKEN 32

template <int block_size, mmq_q8_1_ds_layout ds_layout, bool write_f32>
__launch_bounds__(block_size, 1)
static __global__ void rms_norm_mul_quantize_mmq_q8_1_mmid_kernel(
        const float   * __restrict__ x,
        const float   * __restrict__ mul,
        const int32_t * __restrict__ ids_src1,
        void          * __restrict__ vy,
        float         * __restrict__ dst_f32,
        const int     n_routes,
        const int     ncols,
        const int     ncols_padded,
        const int64_t stride_row,
        const int64_t mul_stride_row,
        const uint3   mul_ncols_packed,
        const uint3   mul_nrows_packed,
        const float   eps) {
    const int token = blockIdx.x;
    const int tid   = threadIdx.x;

    // Shared memory: [0, 32) reduction scratch ; [32, 32+MAX+1) route list + counter.
    extern __shared__ float s_shared_f32[];
    float * s_sum      = s_shared_f32;
    int   * s_routes   = (int *) (s_shared_f32 + 32);
    int   * s_nroutes  = s_routes + MMID_MAX_ROUTES_PER_TOKEN;

    if (tid == 0) { *s_nroutes = 0; }
    __syncthreads();

    // Block-cooperative scan: collect all route indices whose source token matches ours.
    for (int r = tid; r < n_routes; r += block_size) {
        if (ids_src1[r] == token) {
            int slot = atomicAdd(s_nroutes, 1);
            if (slot < MMID_MAX_ROUTES_PER_TOKEN) {
                s_routes[slot] = r;
            }
        }
    }
    __syncthreads();

    const int k_routes = min(*s_nroutes, MMID_MAX_ROUTES_PER_TOKEN);
    if (k_routes == 0) {
        // This token is not routed anywhere; nothing to do.
        return;
    }

    // --- Phase 1: RMS reduction on the source-token row ---
    const float * x_row = x + (int64_t) token * stride_row;
    float sumsq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x_row[col];
        sumsq += xi * xi;
    }
    sumsq = block_reduce<block_reduce_method::SUM, block_size>(sumsq, s_sum);
    const float rms_scale = rsqrtf(sumsq / ncols + eps);

    // --- Phase 2+3: normalize, multiply, quantize, fan out Q8_1 ---
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    float * dst_row = nullptr;
    if constexpr (write_f32) {
        dst_row = dst_f32 + (int64_t) token * ncols;
    }

    // Weight is 1D along the column dim in the standard RMS norm op chain.
    const float * mul_row = mul;

    for (int i0 = tid * 4; i0 < ncols_padded; i0 += block_size * 4) {
        float4 xi;
        if (i0 + 3 < ncols) {
            xi = *reinterpret_cast<const float4 *>(&x_row[i0]);
        } else {
            xi.x = (i0 + 0 < ncols) ? x_row[i0 + 0] : 0.0f;
            xi.y = (i0 + 1 < ncols) ? x_row[i0 + 1] : 0.0f;
            xi.z = (i0 + 2 < ncols) ? x_row[i0 + 2] : 0.0f;
            xi.w = (i0 + 3 < ncols) ? x_row[i0 + 3] : 0.0f;
        }

        const uint32_t mcol0 = fastmodulo((uint32_t)(i0 + 0), mul_ncols_packed);
        const uint32_t mcol1 = fastmodulo((uint32_t)(i0 + 1), mul_ncols_packed);
        const uint32_t mcol2 = fastmodulo((uint32_t)(i0 + 2), mul_ncols_packed);
        const uint32_t mcol3 = fastmodulo((uint32_t)(i0 + 3), mul_ncols_packed);
        xi.x = rms_scale * xi.x * mul_row[mcol0];
        xi.y = rms_scale * xi.y * mul_row[mcol1];
        xi.z = rms_scale * xi.z * mul_row[mcol2];
        xi.w = rms_scale * xi.w * mul_row[mcol3];

        if constexpr (write_f32) {
            if (i0 + 3 < ncols) {
                *reinterpret_cast<float4 *>(&dst_row[i0]) = xi;
            } else {
                if (i0 + 0 < ncols) dst_row[i0 + 0] = xi.x;
                if (i0 + 1 < ncols) dst_row[i0 + 1] = xi.y;
                if (i0 + 2 < ncols) dst_row[i0 + 2] = xi.z;
                if (i0 + 3 < ncols) dst_row[i0 + 3] = xi.w;
            }
        }

        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
        }

        float sum = 0.0f;
        if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
            sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
            for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
            }
        }

        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x * d_inv);
        q.y = roundf(xi.y * d_inv);
        q.z = roundf(xi.z * d_inv);
        q.w = roundf(xi.w * d_inv);
        const float d = 1.0f / d_inv;
        const int iqs = i0 % (4 * QK8_1);

        // Fan out: write the same q/d/sum to every route output whose source is this token.
        for (int k = 0; k < k_routes; ++k) {
            const int r = s_routes[k];
            const int64_t ib = (int64_t)(i0 / (4 * QK8_1)) * n_routes + r;

            char4 * yqs4 = (char4 *) y[ib].qs;
            yqs4[iqs / 4] = q;

            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
                if ((iqs % 16) == 0 && iqs < 96) {
                    y[ib].d2s6[2 + iqs / 16] = sum;
                }
                if ((iqs % 64) == 0) {
                    y[ib].d2s6[iqs / 64] = d;
                }
                continue;
            }
            if ((iqs % 32) != 0) {
                continue;
            }
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs / 32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs / 32] = d;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Fused RMS_NORM + MUL + quantize-to-Q8_1 (block_q8_1 layout, for MMVQ path)
// -----------------------------------------------------------------------------

template <int block_size, bool write_f32>
__launch_bounds__(block_size, 1)
static __global__ void rms_norm_mul_quantize_row_q8_1_kernel(
        const float * __restrict__ x,
        const float * __restrict__ mul,
        void        * __restrict__ vy,
        float       * __restrict__ dst_f32,
        const int     ncols,
        const int     ncols_padded,
        const int64_t stride_row,
        const int64_t stride_channel,
        const int64_t stride_sample,
        const int64_t mul_stride_row,
        const int64_t mul_stride_channel,
        const int64_t mul_stride_sample,
        const uint3   mul_ncols_packed,
        const uint3   mul_nrows_packed,
        const uint3   mul_nchannels_packed,
        const uint3   mul_nsamples_packed,
        const float   eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    const float * x_row = x + sample * stride_sample + channel * stride_channel + row * stride_row;

    const uint32_t mr = fastmodulo(row,     mul_nrows_packed);
    const uint32_t mc = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t ms = fastmodulo(sample,  mul_nsamples_packed);
    const float * mul_row = mul + ms * mul_stride_sample + mc * mul_stride_channel + mr * mul_stride_row;

    // ---- Phase 1: sum of squares over the (unpadded) row ----
    float sumsq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x_row[col];
        sumsq += xi * xi;
    }

    extern __shared__ float s_sum[];
    sumsq = block_reduce<block_reduce_method::SUM, block_size>(sumsq, s_sum);

    const float rms_scale = rsqrtf(sumsq / ncols + eps);

    // ---- Phase 2+3: normalize, multiply, quantize, write ----
    // Each block_q8_1 packs 32 values. A warp (32 lanes × 1 val per lane) fills exactly one block.
    // Layout matches quantize_q8_1: i_cont = ((sample*nchannels + channel)*nrows + row)*ncols_padded + col,
    // ib = i_cont / QK8_1, iqs = i_cont % QK8_1.
    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t batch_flat = (int64_t) sample * nchannels + channel;
    const int64_t row_base   = (batch_flat * nrows + row) * (ncols_padded / QK8_1);

    float * dst_row = nullptr;
    if constexpr (write_f32) {
        dst_row = dst_f32 + (batch_flat * nrows + row) * ncols;
    }

    for (int col = tid; col < ncols_padded; col += block_size) {
        const float xi_raw = (col < ncols) ? x_row[col] : 0.0f;
        const uint32_t mcol = fastmodulo((uint32_t) col, mul_ncols_packed);
        const float xi = rms_scale * xi_raw * ((col < ncols) ? mul_row[mcol] : 0.0f);

        if constexpr (write_f32) {
            if (col < ncols) {
                dst_row[col] = xi;
            }
        }

        float amax = fabsf(xi);
        float sum  = xi;

        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);

        const float  d = amax / 127.0f;
        const int8_t q = (amax == 0.0f) ? 0 : (int8_t) roundf(xi / d);

        const int64_t ib  = row_base + (col / QK8_1);
        const int     iqs = col % QK8_1;

        y[ib].qs[iqs] = q;
        if (iqs == 0) {
            y[ib].ds = make_half2(d, sum);
        }
    }
}

// -----------------------------------------------------------------------------
// Host launchers
// -----------------------------------------------------------------------------

static void rms_norm_mul_quantize_mmq_q8_1_launch(
        mmq_q8_1_ds_layout ds_layout,
        const float * x, const float * mul, const int32_t * ids, void * vy, float * dst_f32,
        int ncols, int ncols_padded, int nrows, int nchannels, int nsamples,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample,
        int64_t mul_stride_row, int64_t mul_stride_channel, int64_t mul_stride_sample,
        uint32_t mul_ncols, uint32_t mul_nrows, uint32_t mul_nchannels, uint32_t mul_nsamples,
        float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);

    const uint3 mul_ncols_p     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_p     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_p = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_p  = init_fastdiv_values(mul_nsamples);

    // Match rms_norm_f32_cuda's block-size heuristic.
    const int block_size = (ncols < 1024) ? 256 : 1024;
    const size_t shmem   = (block_size > WARP_SIZE) ? 32 * sizeof(float) : 0;
    const bool do_f32    = dst_f32 != nullptr;
    const bool do_ids    = ids != nullptr;

#define LAUNCH_FUSED_MMQ(BS, DSL, WF32, IDS)                                                                        \
    rms_norm_mul_quantize_mmq_q8_1_kernel<BS, DSL, WF32, IDS><<<blocks_num, dim3(BS, 1, 1), shmem, stream>>>(       \
            x, mul, ids, vy, dst_f32, ncols, ncols_padded,                                                          \
            stride_row, stride_channel, stride_sample,                                                              \
            mul_stride_row, mul_stride_channel, mul_stride_sample,                                                  \
            mul_ncols_p, mul_nrows_p, mul_nchannels_p, mul_nsamples_p, eps)

#define LAUNCH_FUSED_MMQ_IDS(BS, DSL, WF32)                                                                         \
    if (do_ids) { LAUNCH_FUSED_MMQ(BS, DSL, WF32, true); }                                                          \
    else        { LAUNCH_FUSED_MMQ(BS, DSL, WF32, false); }

#define LAUNCH_FUSED_MMQ_DSL(BS, DSL)                                                                               \
    if (do_f32) { LAUNCH_FUSED_MMQ_IDS(BS, DSL, true); }                                                            \
    else        { LAUNCH_FUSED_MMQ_IDS(BS, DSL, false); }

    switch (ds_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            if (block_size == 256) { LAUNCH_FUSED_MMQ_DSL(256,  MMQ_Q8_1_DS_LAYOUT_D4); }
            else                   { LAUNCH_FUSED_MMQ_DSL(1024, MMQ_Q8_1_DS_LAYOUT_D4); }
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            if (block_size == 256) { LAUNCH_FUSED_MMQ_DSL(256,  MMQ_Q8_1_DS_LAYOUT_DS4); }
            else                   { LAUNCH_FUSED_MMQ_DSL(1024, MMQ_Q8_1_DS_LAYOUT_DS4); }
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            if (block_size == 256) { LAUNCH_FUSED_MMQ_DSL(256,  MMQ_Q8_1_DS_LAYOUT_D2S6); }
            else                   { LAUNCH_FUSED_MMQ_DSL(1024, MMQ_Q8_1_DS_LAYOUT_D2S6); }
            break;
    }
#undef LAUNCH_FUSED_MMQ_DSL
#undef LAUNCH_FUSED_MMQ_IDS
#undef LAUNCH_FUSED_MMQ
}

void rms_norm_mul_quantize_mmq_q8_1_cuda(
        const float * x, const float * mul, const int32_t * ids, void * vy, float * dst_f32,
        int ncols, int ncols_padded, int nrows, int nchannels, int nsamples,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample,
        int64_t mul_stride_row, int64_t mul_stride_channel, int64_t mul_stride_sample,
        uint32_t mul_ncols, uint32_t mul_nrows, uint32_t mul_nchannels, uint32_t mul_nsamples,
        float eps, ggml_type type_src0, cudaStream_t stream) {
    GGML_ASSERT(ncols_padded % (4 * QK8_1) == 0);
    rms_norm_mul_quantize_mmq_q8_1_launch(
            mmq_get_q8_1_ds_layout(type_src0),
            x, mul, ids, vy, dst_f32, ncols, ncols_padded, nrows, nchannels, nsamples,
            stride_row, stride_channel, stride_sample,
            mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols, mul_nrows, mul_nchannels, mul_nsamples, eps, stream);
}

void rms_norm_mul_quantize_row_q8_1_cuda(
        const float * x, const float * mul, void * vy, float * dst_f32,
        int ncols, int ncols_padded, int nrows, int nchannels, int nsamples,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample,
        int64_t mul_stride_row, int64_t mul_stride_channel, int64_t mul_stride_sample,
        uint32_t mul_ncols, uint32_t mul_nrows, uint32_t mul_nchannels, uint32_t mul_nsamples,
        float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols_padded % QK8_1 == 0);

    const dim3 blocks_num(nrows, nchannels, nsamples);
    const int block_size = 256;
    const size_t shmem   = 32 * sizeof(float);

    const uint3 mul_ncols_p     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_p     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_p = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_p  = init_fastdiv_values(mul_nsamples);

    if (dst_f32 != nullptr) {
        rms_norm_mul_quantize_row_q8_1_kernel<256, true><<<blocks_num, dim3(block_size, 1, 1), shmem, stream>>>(
                x, mul, vy, dst_f32, ncols, ncols_padded,
                stride_row, stride_channel, stride_sample,
                mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_p, mul_nrows_p, mul_nchannels_p, mul_nsamples_p, eps);
    } else {
        rms_norm_mul_quantize_row_q8_1_kernel<256, false><<<blocks_num, dim3(block_size, 1, 1), shmem, stream>>>(
                x, mul, vy, nullptr, ncols, ncols_padded,
                stride_row, stride_channel, stride_sample,
                mul_stride_row, mul_stride_channel, mul_stride_sample,
                mul_ncols_p, mul_nrows_p, mul_nchannels_p, mul_nsamples_p, eps);
    }
}

// Host launcher for the per-token MMID variant.
static void rms_norm_mul_quantize_mmq_q8_1_mmid_cuda(
        mmq_q8_1_ds_layout ds_layout,
        const float * x, const float * mul, const int32_t * ids_src1,
        void * vy, float * dst_f32,
        int n_tokens, int n_routes,
        int ncols, int ncols_padded,
        int64_t stride_row, int64_t mul_stride_row,
        uint32_t mul_ncols, uint32_t mul_nrows,
        float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols_padded % (4 * QK8_1) == 0);

    const dim3 blocks_num(n_tokens, 1, 1);
    const int  block_size = (ncols < 1024) ? 256 : 1024;
    const size_t shmem    = (size_t) 32 * sizeof(float) +
                            (size_t) MMID_MAX_ROUTES_PER_TOKEN * sizeof(int) +
                            sizeof(int);
    const bool do_f32 = dst_f32 != nullptr;

    const uint3 mul_ncols_p = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_p = init_fastdiv_values(mul_nrows);

#define LAUNCH_MMID(BS, DSL, WF32)                                                                                  \
    rms_norm_mul_quantize_mmq_q8_1_mmid_kernel<BS, DSL, WF32>                                                       \
        <<<blocks_num, dim3(BS, 1, 1), shmem, stream>>>(                                                            \
            x, mul, ids_src1, vy, dst_f32, n_routes, ncols, ncols_padded,                                           \
            stride_row, mul_stride_row, mul_ncols_p, mul_nrows_p, eps)

#define LAUNCH_MMID_DSL(BS, DSL)                                                                                    \
    if (do_f32) { LAUNCH_MMID(BS, DSL, true); }                                                                     \
    else        { LAUNCH_MMID(BS, DSL, false); }

    switch (ds_layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            if (block_size == 256) { LAUNCH_MMID_DSL(256,  MMQ_Q8_1_DS_LAYOUT_D4); }
            else                   { LAUNCH_MMID_DSL(1024, MMQ_Q8_1_DS_LAYOUT_D4); }
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            if (block_size == 256) { LAUNCH_MMID_DSL(256,  MMQ_Q8_1_DS_LAYOUT_DS4); }
            else                   { LAUNCH_MMID_DSL(1024, MMQ_Q8_1_DS_LAYOUT_DS4); }
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            if (block_size == 256) { LAUNCH_MMID_DSL(256,  MMQ_Q8_1_DS_LAYOUT_D2S6); }
            else                   { LAUNCH_MMID_DSL(1024, MMQ_Q8_1_DS_LAYOUT_D2S6); }
            break;
    }
#undef LAUNCH_MMID_DSL
#undef LAUNCH_MMID
}

// -----------------------------------------------------------------------------
// Dispatch: allocate Q8_1 scratch, launch fused kernel, hand off to MMQ / MMVQ.
// -----------------------------------------------------------------------------

void ggml_cuda_op_rms_norm_mul_mat_fused(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm,
        ggml_tensor * mul,
        ggml_tensor * mul_mat) {
    // rms_norm->src[0] is the normalized tensor.
    const ggml_tensor * rms_src = rms_norm->src[0];

    // Identify which mul operand is the weight (the "other" one) vs. the RMS output.
    const ggml_tensor * weight_src = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
    GGML_ASSERT(weight_src != nullptr);

    // src0 = quantized weight matrix of mul_mat; src1 = the mul result (RMS*weight).
    // For MUL_MAT_ID, src[2] holds the routing ids.
    const ggml_tensor * mm_src0 = mul_mat->src[0];
    const ggml_tensor * mm_src1 = mul_mat->src[1];
    const ggml_tensor * mm_ids  = (mul_mat->op == GGML_OP_MUL_MAT_ID) ? mul_mat->src[2] : nullptr;
    GGML_ASSERT(mm_src1 == mul);

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int64_t ts_src = ggml_type_size(rms_src->type);
    GGML_ASSERT(ts_src == sizeof(float));
    const int64_t stride_row     = rms_src->nb[1] / ts_src;
    const int64_t stride_channel = rms_src->nb[2] / ts_src;
    const int64_t stride_sample  = rms_src->nb[3] / ts_src;

    const int64_t ts_mul = ggml_type_size(weight_src->type);
    GGML_ASSERT(ts_mul == sizeof(float));
    const int64_t mul_stride_row     = weight_src->nb[1] / ts_mul;
    const int64_t mul_stride_channel = weight_src->nb[2] / ts_mul;
    const int64_t mul_stride_sample  = weight_src->nb[3] / ts_mul;

    const int ncols     = (int) rms_src->ne[0];
    const int nrows     = (int) rms_src->ne[1];
    const int nchannels = (int) rms_src->ne[2];
    const int nsamples  = (int) rms_src->ne[3];

    const int64_t ne10 = mm_src1->ne[0];
    const int64_t ne11 = mm_src1->ne[1];
    const int64_t ne12 = mm_src1->ne[2];
    const int64_t ne13 = mm_src1->ne[3];
    GGML_ASSERT(ne10 == ncols);
    GGML_ASSERT(ne11 == nrows);
    GGML_ASSERT(ne12 == nchannels);
    GGML_ASSERT(ne13 == nsamples);

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    GGML_ASSERT(ne10_padded < (1LL << 31));

    // Always write the MUL tensor's F32 output — it may have other consumers or be a
    // graph output, and the bandwidth cost is the same as the unfused RMS+MUL kernel.
    float * dst_f32 = (float *) mul->data;

    // ---- MUL_MAT_ID path: routed quantize + MMQ, or unrouted quantize + MMVQ ----
    if (mm_ids) {
        const int64_t n_expert_used = mm_ids->ne[0];
        const int64_t ne02          = mm_src0->ne[2];

        // Priority: MMVQ if ne12 is small enough (matches ggml_cuda_mul_mat_id).
        const bool use_vec_q_id = ggml_is_quantized(mm_src0->type) && ne12 <= MMVQ_MAX_BATCH_SIZE &&
                                  ne12 <= get_mmvq_mmid_max_batch(mm_src0->type, cc);
        const bool use_q_id     = ggml_is_quantized(mm_src0->type) &&
                                  ggml_cuda_should_use_mmq(mm_src0->type, cc, ne12, /*n_experts=*/ne02);

        if (use_vec_q_id) {
            // MMVQ for MUL_MAT_ID: quantize the full src1 (no ids), let MMVQ route internally.
            const size_t nbytes_src1_q8_1 = ne13 * ne12 * ne11 * ne10_padded *
                                            sizeof(block_q8_1) / QK8_1;
            ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

            rms_norm_mul_quantize_row_q8_1_cuda(
                    (const float *) rms_src->data,
                    (const float *) weight_src->data,
                    src1_q8_1.get(), dst_f32,
                    ncols, (int) ne10_padded, nrows, nchannels, nsamples,
                    stride_row, stride_channel, stride_sample,
                    mul_stride_row, mul_stride_channel, mul_stride_sample,
                    (uint32_t) weight_src->ne[0], (uint32_t) weight_src->ne[1],
                    (uint32_t) weight_src->ne[2], (uint32_t) weight_src->ne[3],
                    eps, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.src1_q8_1_pre = src1_q8_1.get();
            ggml_cuda_mul_mat_vec_q(ctx, mm_src0, mm_src1, mm_ids, mul_mat, &fusion);
            return;
        }

        GGML_ASSERT(use_q_id);

        // MMQ for MUL_MAT_ID: produce PRE-ROUTED Q8_1 (one output per (token, expert) pair).
        // We need ids_src1 for the fused kernel's routing lookup. ggml_cuda_mul_mat_q will
        // also build its own ids_src1 / ids_dst / expert_bounds; that's a small redundancy
        // (O(n_tokens * n_expert_used) int32s) we accept in exchange for leaving the MMQ
        // dispatcher untouched.
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(mm_ids->nb[0] == ggml_element_size(mm_ids));

        const int64_t ne_get_rows = ne12 * n_expert_used;
        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst_scratch(ctx.pool(), ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> expert_bounds_scratch(ctx.pool(), ne02 + 1);

        const int si1  = mm_ids->nb[1] / ggml_element_size(mm_ids);
        const int sis1 = (int)(mm_src1->nb[2] / mm_src1->nb[1]);
        ggml_cuda_launch_mm_ids_helper((const int32_t *) mm_ids->data, ids_src1.get(),
                ids_dst_scratch.get(), expert_bounds_scratch.get(),
                (int) ne02, (int) ne12, (int) n_expert_used, (int) ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());

        const size_t nbytes_src1_q8_1 =
            ne12 * n_expert_used * ne10_padded * sizeof(block_q8_1) / QK8_1 +
            get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        const int64_t ne11_flat = ne12 * n_expert_used;

        // Per-token fused kernel: RMS+MUL+quantize computed once per token, Q8_1 data
        // fanned out to all routes owning that token. For MUL_MAT_ID the source-token
        // dimension in src1 is stored at index 2 (ne12 == n_tokens), and the src1 row
        // stride is nb[2] / sizeof(float).
        const int64_t ts_src1 = ggml_type_size(mm_src1->type);
        const int64_t src1_stride_token = mm_src1->nb[2] / ts_src1;

        rms_norm_mul_quantize_mmq_q8_1_mmid_cuda(
                mmq_get_q8_1_ds_layout(mm_src0->type),
                (const float *) rms_src->data,
                (const float *) weight_src->data,
                ids_src1.get(),
                src1_q8_1.get(), dst_f32,
                (int) ne12, (int) ne11_flat,
                ncols, (int) ne10_padded,
                src1_stride_token, mul_stride_row,
                (uint32_t) weight_src->ne[0], (uint32_t) weight_src->ne[1],
                eps, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_mm_fusion_args_host fusion{};
        fusion.src1_q8_1_pre = src1_q8_1.get();
        ggml_cuda_mul_mat_q(ctx, mm_src0, mm_src1, mm_ids, mul_mat, &fusion);
        return;
    }

    // ---- Plain MUL_MAT path ----
    const bool use_mul_mat_vec_q =
        ggml_is_quantized(mm_src0->type) && ne11 <= MMVQ_MAX_BATCH_SIZE;
    const bool use_mul_mat_q =
        ggml_is_quantized(mm_src0->type) &&
        ggml_cuda_should_use_mmq(mm_src0->type, cc, ne11, /*n_experts=*/0);

    if (use_mul_mat_vec_q) {
        const size_t nbytes_src1_q8_1 = ne13 * ne12 * ne11 * ne10_padded *
                                        sizeof(block_q8_1) / QK8_1;
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        rms_norm_mul_quantize_row_q8_1_cuda(
                (const float *) rms_src->data,
                (const float *) weight_src->data,
                src1_q8_1.get(), dst_f32,
                ncols, (int) ne10_padded, nrows, nchannels, nsamples,
                stride_row, stride_channel, stride_sample,
                mul_stride_row, mul_stride_channel, mul_stride_sample,
                (uint32_t) weight_src->ne[0], (uint32_t) weight_src->ne[1],
                (uint32_t) weight_src->ne[2], (uint32_t) weight_src->ne[3],
                eps, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_mm_fusion_args_host fusion{};
        fusion.src1_q8_1_pre = src1_q8_1.get();
        ggml_cuda_mul_mat_vec_q(ctx, mm_src0, mm_src1, nullptr, mul_mat, &fusion);
        return;
    }

    GGML_ASSERT(use_mul_mat_q);

    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    rms_norm_mul_quantize_mmq_q8_1_cuda(
            (const float *) rms_src->data,
            (const float *) weight_src->data,
            /*ids=*/nullptr,
            src1_q8_1.get(), dst_f32,
            ncols, (int) ne10_padded, nrows, nchannels, nsamples,
            stride_row, stride_channel, stride_sample,
            mul_stride_row, mul_stride_channel, mul_stride_sample,
            (uint32_t) weight_src->ne[0], (uint32_t) weight_src->ne[1],
            (uint32_t) weight_src->ne[2], (uint32_t) weight_src->ne[3],
            eps, mm_src0->type, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.src1_q8_1_pre = src1_q8_1.get();
    ggml_cuda_mul_mat_q(ctx, mm_src0, mm_src1, nullptr, mul_mat, &fusion);
}
