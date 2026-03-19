#include "fused-softplus.cuh"

#define CUDA_FUSED_SOFTPLUS_BLOCK_SIZE 256

static __global__ void fused_softplus_kernel(
    const float * __restrict__ x, const float * __restrict__ add_a, const float * __restrict__ add_b, float * __restrict__ dst,
    const int64_t ne, const uint3 ne0,
    const int64_t x_stride, const int64_t a_stride, const int64_t b_stride) {

    for (int64_t i = blockDim.x * blockIdx.x + threadIdx.x; i < ne; i += blockDim.x * gridDim.x) {
        const uint2 dm = fast_div_modulo((uint32_t)i, ne0);
        const int64_t j1 = dm.x;
        const int64_t j0 = dm.y;

        const float a_val = add_a[j1 * a_stride + j0];
        const float b_val = add_b[j1 * b_stride + j0];
        const float x_val = x[j1 * x_stride + j0];

        const float sum = a_val + b_val;
        const float sp  = (sum > 20.0f) ? sum : logf(1.0f + expf(sum));

        dst[i] = sp * x_val;
    }
}

// softplus(add_src0 + add_src1) * x, where x is the non-softplus input to the MUL node
void ggml_cuda_op_fused_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * add_src0, ggml_tensor * add_src1) {
    const ggml_tensor * x = (dst->src[0]->op == GGML_OP_UNARY) ? dst->src[1] : dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(x));
    GGML_ASSERT(ggml_is_contiguous_1(add_src0));
    GGML_ASSERT(ggml_is_contiguous_1(add_src1));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    const int64_t ne     = ggml_nelements(dst);
    const uint3   ne0_fd = init_fastdiv_values(dst->ne[0]);

    const int64_t max_blocks = 256;
    const int64_t num_blocks = std::min((ne + CUDA_FUSED_SOFTPLUS_BLOCK_SIZE - 1) / CUDA_FUSED_SOFTPLUS_BLOCK_SIZE, max_blocks);

    auto bcast_stride = [](const ggml_tensor * t) -> int64_t {
        return (t->ne[1] > 1) ? (t->nb[1] / sizeof(float)) : 0;
    };

    fused_softplus_kernel<<<num_blocks, CUDA_FUSED_SOFTPLUS_BLOCK_SIZE, 0, stream>>>(
        (const float *)x->data, (const float *)add_src0->data, (const float *)add_src1->data,
        (float *)dst->data, ne, ne0_fd,
        bcast_stride(x), bcast_stride(add_src0), bcast_stride(add_src1));
}
