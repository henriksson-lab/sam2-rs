#include <tuple>
#include <cuda/mem_eff_attention/kernel_forward.h>
#include <cuda_runtime.h>

using namespace PyTorchMemEffAttention;

using Kernel64 = AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>;
using Kernel128 = AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>;

template <typename Kernel>
__global__ void sam2_mem_eff_kernel(typename Kernel::Params params) {
    if (params.advance_to_block()) {
        Kernel::attention_kernel(params);
    }
}

template <typename Kernel>
int launch_typed(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch,
    int q_sequence,
    int kv_sequence,
    int heads,
    int head_dim,
    long long q_stride_b,
    int q_stride_m,
    int q_stride_h,
    long long k_stride_b,
    int k_stride_m,
    int k_stride_h,
    long long v_stride_b,
    int v_stride_m,
    int v_stride_h,
    float scale,
    cudaStream_t stream) {
    typename Kernel::Params p{};
    p.query_ptr = q;
    p.key_ptr = k;
    p.value_ptr = v;
    p.output_ptr = output;
    p.num_batches = batch;
    p.num_heads = heads;
    p.q_heads_per_kv = 1;
    p.head_dim = head_dim;
    p.head_dim_value = head_dim;
    p.num_queries = q_sequence;
    p.num_keys = kv_sequence;
    p.num_keys_absolute = kv_sequence;
    p.scale = scale;
    p.q_strideB = q_stride_b;
    p.q_strideM = q_stride_m;
    p.q_strideH = q_stride_h;
    p.k_strideB = k_stride_b;
    p.k_strideM = k_stride_m;
    p.k_strideH = k_stride_h;
    p.v_strideB = v_stride_b;
    p.v_strideM = v_stride_m;
    p.v_strideH = v_stride_h;
    p.o_strideM = heads * head_dim;
    const size_t shared = sizeof(typename Kernel::SharedStorage);
    if (shared > 0xc000) {
        auto error = cudaFuncSetAttribute(
            sam2_mem_eff_kernel<Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared);
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    sam2_mem_eff_kernel<Kernel><<<p.getBlocksGrid(), p.getThreadsGrid(), shared, stream>>>(p);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int sam2_mem_eff_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch,
    int q_sequence,
    int kv_sequence,
    int heads,
    int head_dim,
    long long q_stride_b,
    int q_stride_m,
    int q_stride_h,
    long long k_stride_b,
    int k_stride_m,
    int k_stride_h,
    long long v_stride_b,
    int v_stride_m,
    int v_stride_h,
    float scale,
    cudaStream_t stream) {
    if (head_dim <= 64) {
        return launch_typed<Kernel64>(q, k, v, output, batch, q_sequence, kv_sequence, heads, head_dim,
            q_stride_b, q_stride_m, q_stride_h, k_stride_b, k_stride_m, k_stride_h,
            v_stride_b, v_stride_m, v_stride_h, scale, stream);
    }
    if (head_dim <= 128) {
        return launch_typed<Kernel128>(q, k, v, output, batch, q_sequence, kv_sequence, heads, head_dim,
            q_stride_b, q_stride_m, q_stride_h, k_stride_b, k_stride_m, k_stride_h,
            v_stride_b, v_stride_m, v_stride_h, scale, stream);
    }
    return -1;
}

__global__ void sam2_max_pool2d_nhwc_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned long long output_elements,
    unsigned int height,
    unsigned int width,
    unsigned int channels,
    unsigned long long stride_b,
    unsigned long long stride_y,
    unsigned long long stride_x) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    const unsigned int channel = index % channels;
    const unsigned long long pixel = index / channels;
    const unsigned int out_width = width / 2;
    const unsigned int x = (pixel % out_width) * 2;
    const unsigned int y = ((pixel / out_width) % (height / 2)) * 2;
    const unsigned long long batch = pixel / ((height / 2) * out_width);
    const unsigned long long source =
        batch * stride_b + (unsigned long long)y * stride_y +
        (unsigned long long)x * stride_x + channel;
    float value = fmaxf(input[source], input[source + stride_x]);
    value = fmaxf(value, input[source + stride_y]);
    value = fmaxf(value, input[source + stride_y + stride_x]);
    output[index] = value;
}

extern "C" int sam2_max_pool2d_nhwc_f32(
    const float* input,
    float* output,
    unsigned long long output_elements,
    unsigned int height,
    unsigned int width,
    unsigned int channels,
    unsigned long long stride_b,
    unsigned long long stride_y,
    unsigned long long stride_x,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    const unsigned int blocks =
        (unsigned int)((output_elements + threads - 1) / threads);
    sam2_max_pool2d_nhwc_f32_kernel<<<blocks, threads, 0, stream>>>(
        input, output, output_elements, height, width, channels,
        stride_b, stride_y, stride_x);
    return static_cast<int>(cudaGetLastError());
}
