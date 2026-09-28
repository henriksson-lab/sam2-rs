#include <tuple>
#include <cuda/mem_eff_attention/kernel_forward.h>
#include <cuda_runtime.h>

using namespace PyTorchMemEffAttention;

using Kernel64 = AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, true, true>;
// Decoder token-to-image attention has only 8 queries. The kernel requires a
// multiple-of-32 query tile, so avoid wasting a second warp-tile there while
// retaining the 64-query configuration for image and temporal attention.
using KernelShortQ = AttentionKernel<float, cutlass::arch::Sm75, true, 32, 64, 64, true, true>;
using Kernel128 = AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128, true, true>;

__device__ __forceinline__ float2 sam2_warp_sum(float2 value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value.x += __shfl_xor_sync(0xffffffff, value.x, mask, 32);
        value.y += __shfl_xor_sync(0xffffffff, value.y, mask, 32);
    }
    return value;
}

__global__ void sam2_layer_norm_residual_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    int columns,
    float eps) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const long long offset = static_cast<long long>(row) * columns;
    float2 mean_var = make_float2(0.f, 0.f);
    for (int column = lane; column < columns; column += 32) {
        const float value = x[offset + column] + residual[offset + column];
        mean_var.x += value;
        mean_var.y += value * value;
    }
    mean_var = sam2_warp_sum(mean_var);
    const float mean = mean_var.x / columns;
    const float variance = mean_var.y / columns - mean * mean;
    const float inv_std = rsqrtf(variance + eps);
    for (int column = lane; column < columns; column += 32) {
        const float value = x[offset + column] + residual[offset + column];
        output[offset + column] =
            (value - mean) * inv_std * weight[column] + bias[column];
    }
}

extern "C" int sam2_layer_norm_residual_f32(
    const float* x,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int columns,
    float eps,
    cudaStream_t stream) {
    sam2_layer_norm_residual_kernel<<<rows, 32, 0, stream>>>(
        x, residual, weight, bias, output, columns, eps);
    return static_cast<int>(cudaGetLastError());
}

__global__ void sam2_bilinear_stability_counts_kernel(
    const float* input,
    unsigned int* counts,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    float high_threshold,
    float low_threshold) {
    const int mask = blockIdx.x;
    const int lane = threadIdx.x;
    const int output_pixels = output_height * output_width;
    const long long input_base =
        static_cast<long long>(mask) * input_height * input_width;
    const float height_scale = static_cast<float>(input_height) / output_height;
    const float width_scale = static_cast<float>(input_width) / output_width;
    unsigned int high = 0;
    unsigned int low = 0;
    for (int pixel = lane; pixel < output_pixels; pixel += blockDim.x) {
        const int output_y = pixel / output_width;
        const int output_x = pixel - output_y * output_width;
        const float source_y = fmaxf(
            0.f, height_scale * (static_cast<float>(output_y) + 0.5f) - 0.5f);
        const float source_x = fmaxf(
            0.f, width_scale * (static_cast<float>(output_x) + 0.5f) - 0.5f);
        const int y0 = static_cast<int>(floorf(source_y));
        const int x0 = static_cast<int>(floorf(source_x));
        const int y1 = min(y0 + 1, input_height - 1);
        const int x1 = min(x0 + 1, input_width - 1);
        const float weight_y = fminf(fmaxf(source_y - y0, 0.f), 1.f);
        const float weight_x = fminf(fmaxf(source_x - x0, 0.f), 1.f);
        const float v00 = input[input_base + y0 * input_width + x0];
        const float v10 = input[input_base + y0 * input_width + x1];
        const float v01 = input[input_base + y1 * input_width + x0];
        const float v11 = input[input_base + y1 * input_width + x1];
        const float top = v00 * (1.f - weight_x) + v10 * weight_x;
        const float bottom = v01 * (1.f - weight_x) + v11 * weight_x;
        const float value = top * (1.f - weight_y) + bottom * weight_y;
        high += value > high_threshold;
        low += value > low_threshold;
    }
    __shared__ unsigned int high_sums[256];
    __shared__ unsigned int low_sums[256];
    high_sums[lane] = high;
    low_sums[lane] = low;
    for (int stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        if (lane < stride) {
            high_sums[lane] += high_sums[lane + stride];
            low_sums[lane] += low_sums[lane + stride];
        }
    }
    if (lane == 0) {
        counts[mask * 2] = high_sums[0];
        counts[mask * 2 + 1] = low_sums[0];
    }
}

extern "C" int sam2_bilinear_stability_counts_f32(
    const float* input,
    unsigned int* counts,
    int masks,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    float high_threshold,
    float low_threshold,
    cudaStream_t stream) {
    sam2_bilinear_stability_counts_kernel<<<masks, 256, 0, stream>>>(
        input,
        counts,
        input_height,
        input_width,
        output_height,
        output_width,
        high_threshold,
        low_threshold);
    return static_cast<int>(cudaGetLastError());
}

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
    if (q_sequence <= 32 && head_dim <= 64) {
        return launch_typed<KernelShortQ>(q, k, v, output, batch, q_sequence, kv_sequence, heads, head_dim,
            q_stride_b, q_stride_m, q_stride_h, k_stride_b, k_stride_m, k_stride_h,
            v_stride_b, v_stride_m, v_stride_h, scale, stream);
    }
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

__global__ void sam2_conv1x1_nchw_f32_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned long long output_elements,
    unsigned int input_channels,
    unsigned int output_channels,
    unsigned int spatial) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    const unsigned int pixel = index % spatial;
    const unsigned long long channel_index = index / spatial;
    const unsigned int output_channel = channel_index % output_channels;
    const unsigned long long batch = channel_index / output_channels;
    float value = bias[output_channel];
    const unsigned long long input_base = batch * input_channels * spatial + pixel;
    const unsigned long long weight_base =
        (unsigned long long)output_channel * input_channels;
    for (unsigned int input_channel = 0; input_channel < input_channels; ++input_channel) {
        value = fmaf(
            input[input_base + (unsigned long long)input_channel * spatial],
            weight[weight_base + input_channel],
            value);
    }
    output[index] = value;
}

extern "C" int sam2_conv1x1_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    unsigned long long output_elements,
    unsigned int input_channels,
    unsigned int output_channels,
    unsigned int spatial,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    const unsigned int blocks =
        (unsigned int)((output_elements + threads - 1) / threads);
    sam2_conv1x1_nchw_f32_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, output_elements, input_channels,
        output_channels, spatial);
    return static_cast<int>(cudaGetLastError());
}

__global__ void sam2_conv_transpose2x2_nchw_f32_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    const float* residual,
    float* output,
    unsigned long long output_elements,
    unsigned int input_channels,
    unsigned int output_channels,
    unsigned int input_height,
    unsigned int input_width,
    unsigned long long residual_batch_stride,
    bool gelu) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    const unsigned int output_width = input_width * 2;
    const unsigned int output_height = input_height * 2;
    const unsigned int output_spatial = output_height * output_width;
    const unsigned int output_pixel = index % output_spatial;
    const unsigned long long channel_index = index / output_spatial;
    const unsigned int output_channel = channel_index % output_channels;
    const unsigned long long batch = channel_index / output_channels;
    const unsigned int output_y = output_pixel / output_width;
    const unsigned int output_x = output_pixel % output_width;
    const unsigned int input_pixel =
        (output_y / 2) * input_width + output_x / 2;
    const unsigned int kernel_y = output_y & 1;
    const unsigned int kernel_x = output_x & 1;
    const unsigned int input_spatial = input_height * input_width;
    const unsigned long long input_base = batch * input_channels * input_spatial;
    float value = 0.0f;
    for (unsigned int input_channel = 0; input_channel < input_channels; ++input_channel) {
        const unsigned long long weight_index =
            (((unsigned long long)input_channel * output_channels + output_channel) * 2 +
             kernel_y) * 2 + kernel_x;
        value = fmaf(
            input[input_base + (unsigned long long)input_channel * input_spatial + input_pixel],
            weight[weight_index],
            value);
    }
    value += bias[output_channel];
    value += residual[
        batch * residual_batch_stride +
        (unsigned long long)output_channel * output_spatial + output_pixel];
    if (gelu) value *= normcdff(value);
    output[index] = value;
}

extern "C" int sam2_conv_transpose2x2_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    const float* residual,
    float* output,
    unsigned long long output_elements,
    unsigned int input_channels,
    unsigned int output_channels,
    unsigned int input_height,
    unsigned int input_width,
    unsigned long long residual_batch_stride,
    bool gelu,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    const unsigned int blocks =
        (unsigned int)((output_elements + threads - 1) / threads);
    sam2_conv_transpose2x2_nchw_f32_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, residual, output, output_elements, input_channels,
        output_channels, input_height, input_width, residual_batch_stride, gelu);
    return static_cast<int>(cudaGetLastError());
}

__global__ void sam2_threshold_pack_u32_f32_kernel(
    const float* input,
    unsigned int* output,
    unsigned long long pixels_per_mask,
    unsigned long long words_per_mask,
    unsigned long long output_words,
    float threshold) {
    const unsigned long long output_index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= output_words) return;
    const unsigned long long mask = output_index / words_per_mask;
    const unsigned long long word = output_index % words_per_mask;
    const unsigned long long first = word * 32;
    unsigned int packed = 0;
    #pragma unroll
    for (unsigned int bit = 0; bit < 32; ++bit) {
        const unsigned long long pixel = first + bit;
        if (pixel < pixels_per_mask &&
            input[mask * pixels_per_mask + pixel] > threshold) {
            packed |= 1u << bit;
        }
    }
    output[output_index] = packed;
}

extern "C" int sam2_threshold_pack_u32_f32(
    const float* input,
    unsigned int* output,
    unsigned long long pixels_per_mask,
    unsigned long long words_per_mask,
    unsigned long long output_words,
    float threshold,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    const unsigned int blocks =
        (unsigned int)((output_words + threads - 1) / threads);
    sam2_threshold_pack_u32_f32_kernel<<<blocks, threads, 0, stream>>>(
        input, output, pixels_per_mask, words_per_mask, output_words, threshold);
    return static_cast<int>(cudaGetLastError());
}
