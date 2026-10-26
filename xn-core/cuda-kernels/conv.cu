#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>
#include<math.h>

// ============================================================================
// Helper functions for type conversions
// ============================================================================

template<typename T> __device__ __forceinline__ float to_float(T v);
template<> __device__ __forceinline__ float to_float(float v) { return v; }
template<> __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
#if __CUDA_ARCH__ >= 800
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
#endif

template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ float from_float(float v) { return v; }
template<> __device__ __forceinline__ __half from_float(float v) { return __float2half(v); }
#if __CUDA_ARCH__ >= 800
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
#endif

// ============================================================================
// Tiled Im2Col1D kernel
// Transforms input from [B, C_in, L_in] to [B, L_out, C_in * K] for conv1d
// Grid: (ceil(C_in*K/TILE), ceil(L_out/TILE), B), Block: (TILE, BLOCK_ROWS)
// threadIdx.x along C_in*K (contiguous in output) → coalesced writes
// ============================================================================

template <typename T, int TILE_DIM = 32, int BLOCK_ROWS = 8>
__device__ void im2col1d_kernel(
    const size_t c_in,
    const size_t l_in,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const T *src,
    T *dst
) {
    const size_t b = blockIdx.z;
    const size_t ck = c_in * k_size;

    int ck_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    int l_base = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int l_idx = l_base + j;
        if (ck_idx < (int)ck && l_idx < (int)l_out) {
            int k_idx = ck_idx % (int)k_size;
            int c_idx = ck_idx / (int)k_size;

            size_t src_l = (size_t)l_idx * stride + (size_t)k_idx * dilation;
            size_t dst_idx = b * l_out * ck + (size_t)l_idx * ck + (size_t)ck_idx;

            if (src_l < padding || src_l >= l_in + padding) {
                dst[dst_idx] = static_cast<T>(0);
            } else {
                src_l -= padding;
                size_t src_idx = b * c_in * l_in + (size_t)c_idx * l_in + src_l;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

// ============================================================================
// Tiled Col2Im1D kernel
// Transforms col data from [B, L_in, C_out * K] to [B, C_out, L_out]
// Grid: (ceil(L_out/TILE), ceil(C_out/TILE), B), Block: (TILE, BLOCK_ROWS)
// threadIdx.x along L_out (contiguous in output [B, C_out, L_out]) → coalesced writes
// ============================================================================

template <typename T, int TILE_DIM = 32, int BLOCK_ROWS = 8>
__device__ void col2im1d_kernel(
    const size_t l_in,
    const size_t c_out,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const T *src,
    T *dst
) {
    const size_t b = blockIdx.z;

    // threadIdx.x along L_out → coalesced writes
    int l_out_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    int c_base = blockIdx.y * TILE_DIM + threadIdx.y;

    // src layout: [B, L_in, C_out, K]
    const size_t src_s0 = l_in * c_out * k_size;
    const size_t src_s1 = c_out * k_size;
    const size_t src_s2 = k_size;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int c_idx = c_base + j;
        if (l_out_idx < (int)l_out && c_idx < (int)c_out) {
            T sum = static_cast<T>(0);

            // Find all (l_in_idx, k_idx) pairs that contribute to this output position
            // l_out_idx = l_in_idx * stride + k_idx
            int l_in_idx = l_out_idx / (int)stride;
            int k_start = l_out_idx - l_in_idx * (int)stride;

            for (; k_start < (int)k_size && l_in_idx >= 0; k_start += (int)stride, --l_in_idx) {
                if (l_in_idx < (int)l_in) {
                    size_t src_idx = b * src_s0 + (size_t)l_in_idx * src_s1 + (size_t)c_idx * src_s2 + (size_t)k_start;
                    sum += src[src_idx];
                }
            }

            size_t dst_idx = b * c_out * l_out + (size_t)c_idx * l_out + (size_t)l_out_idx;
            dst[dst_idx] = sum;
        }
    }
}

// ============================================================================
// Direct Conv1D kernel (naive implementation for fallback)
// src: [B, C_in, L_in], kernel: [C_out, C_in/groups, K], dst: [B, C_out, L_out]
// ============================================================================

template <typename T>
__device__ void conv1d_direct_kernel(
    const size_t dst_numel,
    const size_t batch,
    const size_t c_in,
    const size_t l_in,
    const size_t c_out,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t groups,
    const T *src,
    const T *kernel,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_numel) return;

    // dst layout: [B, C_out, L_out]
    const size_t l_idx = idx % l_out;
    const size_t c_out_idx = (idx / l_out) % c_out;
    const size_t b_idx = idx / (l_out * c_out);

    const size_t c_in_per_group = c_in / groups;
    const size_t c_out_per_group = c_out / groups;
    const size_t g = c_out_idx / c_out_per_group;
    const size_t c_in_start = g * c_in_per_group;

    float sum = 0.0f;

    for (size_t k = 0; k < k_size; ++k) {
        size_t src_l = l_idx * stride + k * dilation;
        if (src_l < padding || src_l >= l_in + padding) {
            continue;
        }
        src_l -= padding;

        for (size_t c = 0; c < c_in_per_group; ++c) {
            const size_t src_c = c_in_start + c;
            const size_t src_idx = b_idx * c_in * l_in + src_c * l_in + src_l;
            // kernel layout: [C_out, C_in/groups, K]
            const size_t k_idx = c_out_idx * c_in_per_group * k_size + c * k_size + k;
            sum += to_float(src[src_idx]) * to_float(kernel[k_idx]);
        }
    }

    dst[idx] = from_float<T>(sum);
}

// ============================================================================
// Direct Conv Transpose 1D kernel (naive implementation for fallback)
// src: [B, C_in, L_in], kernel: [C_in, C_out/groups, K], dst: [B, C_out, L_out]
// ============================================================================

template <typename T>
__device__ void conv_transpose1d_direct_kernel(
    const size_t dst_numel,
    const size_t batch,
    const size_t c_in,
    const size_t l_in,
    const size_t c_out,
    const size_t l_out,
    const size_t k_size,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t groups,
    const T *src,
    const T *kernel,
    T *dst
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_numel) return;

    // The launch checks every size fits 32 bits, so the index arithmetic, which dominates
    // this kernel, can be done in 32 bits.
    const unsigned int i = idx;
    const unsigned int lo = l_out, co = c_out, ci = c_in, li = l_in, ks = k_size;
    const unsigned int st = stride, dl = dilation;

    // dst layout: [B, C_out, L_out]
    const unsigned int l_out_idx = i % lo;
    const unsigned int c_out_idx = (i / lo) % co;
    const unsigned int b_idx = i / (lo * co);

    const unsigned int c_in_per_group = ci / groups;
    const unsigned int c_out_per_group = co / groups;
    const unsigned int g = c_out_idx / c_out_per_group;
    const unsigned int c_in_start = g * c_in_per_group;
    const unsigned int c_out_in_group = c_out_idx % c_out_per_group;

    float sum = 0.0f;

    // out_l = in_l * stride + k * dilation - padding, so tap k reaches this output when
    // out_l + padding - k * dilation is a non-negative multiple of the stride. With no
    // dilation those taps are every stride-th from (out_l + padding) % stride.
    const int base = (int)l_out_idx + (int)padding;
    const int k_first = dl == 1 ? base % (int)st : 0;
    const int k_step = dl == 1 ? (int)st : 1;
    for (int k = k_first; k < (int)ks; k += k_step) {
        const int in_l_offset = base - k * (int)dl;
        if (in_l_offset < 0) break;
        if (in_l_offset % (int)st != 0) continue;
        const unsigned int in_l = in_l_offset / (int)st;
        if (in_l >= li) continue;

        for (unsigned int c = 0; c < c_in_per_group; ++c) {
            const unsigned int src_c = c_in_start + c;
            const unsigned int src_idx = (b_idx * ci + src_c) * li + in_l;
            // kernel layout: [C_in, C_out/groups, K]
            const unsigned int k_idx = (src_c * c_out_per_group + c_out_in_group) * ks + k;
            sum += to_float(src[src_idx]) * to_float(kernel[k_idx]);
        }
    }

    dst[idx] = from_float<T>(sum);
}

// ============================================================================
// Tiled transpose kernel: [B, L, C] -> [B, C, L]
// Uses shared memory to coalesce both reads and writes.
// Grid: (ceil(C/TILE), ceil(L/TILE), B), Block: (TILE, BLOCK_ROWS)
// ============================================================================

template <typename T, int TILE_DIM = 32, int BLOCK_ROWS = 8>
__device__ void transpose_blc_to_bcl(
    const size_t length,
    const size_t channels,
    const T *src,
    T *dst
) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    const size_t b = blockIdx.z;
    const size_t lc = length * channels;

    // Load: src[b, l, c] — threadIdx.x along C (contiguous) → coalesced reads
    int c = blockIdx.x * TILE_DIM + threadIdx.x;
    int l = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (c < (int)channels && (l + j) < (int)length)
            tile[threadIdx.y + j][threadIdx.x] = src[b * lc + (l + j) * channels + c];
    }

    __syncthreads();

    // Store: dst[b, c, l] — threadIdx.x along L (contiguous) → coalesced writes
    int l2 = blockIdx.y * TILE_DIM + threadIdx.x;
    int c2 = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (l2 < (int)length && (c2 + j) < (int)channels)
            dst[b * lc + (c2 + j) * length + l2] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ============================================================================
// Tiled transpose kernel: [B, C, L] -> [B, L, C]
// Uses shared memory to coalesce both reads and writes.
// Grid: (ceil(L/TILE), ceil(C/TILE), B), Block: (TILE, BLOCK_ROWS)
// ============================================================================

template <typename T, int TILE_DIM = 32, int BLOCK_ROWS = 8>
__device__ void transpose_bcl_to_blc(
    const size_t channels,
    const size_t length,
    const T *src,
    T *dst
) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    const size_t b = blockIdx.z;
    const size_t cl = channels * length;

    // Load: src[b, c, l] — threadIdx.x along L (contiguous) → coalesced reads
    int l = blockIdx.x * TILE_DIM + threadIdx.x;
    int c = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (l < (int)length && (c + j) < (int)channels)
            tile[threadIdx.y + j][threadIdx.x] = src[b * cl + (c + j) * length + l];
    }

    __syncthreads();

    // Store: dst[b, l, c] — threadIdx.x along C (contiguous) → coalesced writes
    int c2 = blockIdx.y * TILE_DIM + threadIdx.x;
    int l2 = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (c2 < (int)channels && (l2 + j) < (int)length)
            dst[b * cl + (l2 + j) * channels + c2] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ============================================================================
// Kernel instantiation macros
// ============================================================================

#define IM2COL1D_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void im2col1d_##RUST_NAME( \
    const size_t c_in, \
    const size_t l_in, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    im2col1d_kernel<TYPENAME>(c_in, l_in, l_out, k_size, stride, padding, dilation, src, dst); \
}

#define COL2IM1D_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void col2im1d_##RUST_NAME( \
    const size_t l_in, \
    const size_t c_out, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    col2im1d_kernel<TYPENAME>(l_in, c_out, l_out, k_size, stride, src, dst); \
}

#define CONV1D_DIRECT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void conv1d_direct_##RUST_NAME( \
    const size_t dst_numel, \
    const size_t batch, \
    const size_t c_in, \
    const size_t l_in, \
    const size_t c_out, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t groups, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
    conv1d_direct_kernel<TYPENAME>(dst_numel, batch, c_in, l_in, c_out, l_out, k_size, stride, padding, dilation, groups, src, kernel, dst); \
}

#define CONV_TRANSPOSE1D_DIRECT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void conv_transpose1d_direct_##RUST_NAME( \
    const size_t dst_numel, \
    const size_t batch, \
    const size_t c_in, \
    const size_t l_in, \
    const size_t c_out, \
    const size_t l_out, \
    const size_t k_size, \
    const size_t stride, \
    const size_t padding, \
    const size_t out_padding, \
    const size_t dilation, \
    const size_t groups, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
    conv_transpose1d_direct_kernel<TYPENAME>(dst_numel, batch, c_in, l_in, c_out, l_out, k_size, stride, padding, out_padding, dilation, groups, src, kernel, dst); \
}

#define TRANSPOSE_BLC_BCL_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void transpose_blc_bcl_##RUST_NAME( \
    const size_t length, \
    const size_t channels, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    transpose_blc_to_bcl<TYPENAME>(length, channels, src, dst); \
}

#define TRANSPOSE_BCL_BLC_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void transpose_bcl_blc_##RUST_NAME( \
    const size_t channels, \
    const size_t length, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
    transpose_bcl_to_blc<TYPENAME>(channels, length, src, dst); \
}

#define ALL_CONV_OPS(TYPENAME, RUST_NAME) \
    IM2COL1D_OP(TYPENAME, RUST_NAME) \
    COL2IM1D_OP(TYPENAME, RUST_NAME) \
    CONV1D_DIRECT_OP(TYPENAME, RUST_NAME) \
    CONV_TRANSPOSE1D_DIRECT_OP(TYPENAME, RUST_NAME) \
    TRANSPOSE_BLC_BCL_OP(TYPENAME, RUST_NAME) \
    TRANSPOSE_BCL_BLC_OP(TYPENAME, RUST_NAME)

// ============================================================================
// Instantiate for all supported types
// ============================================================================

#if __CUDA_ARCH__ >= 800
ALL_CONV_OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
ALL_CONV_OPS(__half, f16)
#endif

ALL_CONV_OPS(float, f32)
