#pragma once

#include "common.cuh"
#include "ggml.h"

// Fused RMS_NORM + MUL + quantize-to-Q8_1 dispatch.
// Detects the cgraph pattern { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT } where
// MUL_MAT's src0 is quantized and the MMQ or MMVQ path would be used; produces Q8_1 blocks
// directly into a scratch buffer that is then consumed by MMQ / MMVQ, and (if the MUL
// output has other consumers) also writes the post-norm post-mul F32 values into MUL's
// output tensor.

void ggml_cuda_op_rms_norm_mul_mat_fused(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm,
        ggml_tensor * mul,
        ggml_tensor * mul_mat);

// Low-level launchers. dst_f32 is optional; when non-null, the post-norm post-mul F32
// values are also written there so downstream consumers of the unquantized MUL output
// can still read it.
// When `ids` is non-null, the kernel operates in MoE-routing mode: blockIdx.x iterates
// the flat (token, expert) route index, ids[blockIdx.x] maps back to the source token row,
// and `nrows` must equal the flat route count (ne12 * n_expert_used).
void rms_norm_mul_quantize_mmq_q8_1_cuda(
        const float * x, const float * mul, const int32_t * ids, void * vy, float * dst_f32,
        int ncols, int ncols_padded, int nrows, int nchannels, int nsamples,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample,
        int64_t mul_stride_row, int64_t mul_stride_channel, int64_t mul_stride_sample,
        uint32_t mul_ncols, uint32_t mul_nrows, uint32_t mul_nchannels, uint32_t mul_nsamples,
        float eps, ggml_type type_src0, cudaStream_t stream);

void rms_norm_mul_quantize_row_q8_1_cuda(
        const float * x, const float * mul, void * vy, float * dst_f32,
        int ncols, int ncols_padded, int nrows, int nchannels, int nsamples,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample,
        int64_t mul_stride_row, int64_t mul_stride_channel, int64_t mul_stride_sample,
        uint32_t mul_ncols, uint32_t mul_nrows, uint32_t mul_nchannels, uint32_t mul_nsamples,
        float eps, cudaStream_t stream);
