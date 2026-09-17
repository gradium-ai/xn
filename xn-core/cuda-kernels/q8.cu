// q8_0 weights on the CUDA backend: dequantize-fused matmuls for small m and a
// plain dequantize for the cuBLAS path at larger m.
//
// The weight is a (n, k) matrix split into two streams on upload (see
// cuda_backend/q8.rs):
//   qs:     row j at j * k bytes, one int8 per weight
//   scales: row j at j * (k/32), one f32 per 32-weight block
//
// `gemm_q8_f32_r<MR>` computes dst[i, j] = sum_l lhs[i, l] * W[j, l] for MR
// rows of `lhs` per block. One warp owns one output column j: per 256-k
// chunk each lane takes two packed words (four int8 each, coalesced 128-byte
// loads per warp) issued before the block's MR lhs rows for that chunk are
// staged in shared memory, and the partial sums are reduced with shuffles at
// the end. At
// m == 1 the kernel is bound by the weight stream, so unpacking to f32 and
// using FMAs costs nothing a packed integer dot would save; for m > 1 the MR
// row accumulators let one pass over the weight serve MR outputs.
#include <stdint.h>

#define Q8_WARPS 8

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float4 unpack4(uint32_t w) {
  return make_float4((float)(int8_t)(w & 0xffu), (float)(int8_t)((w >> 8) & 0xffu),
                     (float)(int8_t)((w >> 16) & 0xffu), (float)(int8_t)(w >> 24));
}

// Words (4 weights each) of a row staged per step: 256 k, eight q8_0 blocks,
// two words per lane.
#define Q8_KC 64

#define Q8_THREADS (Q8_WARPS * 32)
#define Q8_NSTAGE ((MR * Q8_KC + Q8_THREADS - 1) / Q8_THREADS)

template <int MR>
__device__ __forceinline__ void
gemm_q8_rows(const float *__restrict__ lhs, const uint8_t *__restrict__ qs,
             const float *__restrict__ scales, float *__restrict__ dst, const int m,
             const int n, const int k) {
  // [row][word] chunk of the lhs block, shared by the block's eight warps so
  // it is read from L2 once per block rather than once per warp.
  __shared__ float4 lsh[MR * Q8_KC];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int j = blockIdx.x * Q8_WARPS + warp;
  const int i0 = blockIdx.y * MR;
  const int kw = k >> 2;
  // Columns past n read the last valid column; their result is dropped.
  const int jj = min(j, n - 1);
  const uint32_t *qrow = reinterpret_cast<const uint32_t *>(qs + (size_t)jj * k);
  const float *srow = scales + (size_t)jj * (k >> 5);
  const float4 *lhs4 = reinterpret_cast<const float4 *>(lhs);
  float acc[MR];
#pragma unroll
  for (int r = 0; r < MR; r++)
    acc[r] = 0.f;
  for (int c0 = 0; c0 < kw; c0 += Q8_KC) {
    // This lane's two words of the weight row, in flight while the lhs
    // block is staged.
    const int wa = c0 + lane;
    const int wb = wa + 32;
    const bool ha = wa < kw;
    const bool hb = wb < kw;
    const uint32_t pa = ha ? qrow[wa] : 0u;
    const uint32_t pb = hb ? qrow[wb] : 0u;
    const float da = ha ? srow[wa >> 3] : 0.f;
    const float db = hb ? srow[wb >> 3] : 0.f;
    // Loads first, stores after, with a fixed trip count so the loads are
    // issued back to back rather than each behind the previous store.
    float4 stage[Q8_NSTAGE];
#pragma unroll
    for (int s = 0; s < Q8_NSTAGE; s++) {
      const int e = tid + s * Q8_THREADS;
      const int r = e / Q8_KC;
      const int q = e % Q8_KC;
      const int row = min(i0 + r, m - 1);
      stage[s] = (e < MR * Q8_KC && c0 + q < kw) ? lhs4[(size_t)row * kw + c0 + q]
                                                  : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int s = 0; s < Q8_NSTAGE; s++) {
      const int e = tid + s * Q8_THREADS;
      if (e < MR * Q8_KC)
        lsh[e] = stage[s];
    }
    __syncthreads();
    if (ha) {
      const float4 q = unpack4(pa);
#pragma unroll
      for (int r = 0; r < MR; r++) {
        const float4 a = lsh[r * Q8_KC + lane];
        acc[r] = fmaf(da, q.x * a.x + q.y * a.y + q.z * a.z + q.w * a.w, acc[r]);
      }
    }
    if (hb) {
      const float4 q = unpack4(pb);
#pragma unroll
      for (int r = 0; r < MR; r++) {
        const float4 a = lsh[r * Q8_KC + lane + 32];
        acc[r] = fmaf(db, q.x * a.x + q.y * a.y + q.z * a.z + q.w * a.w, acc[r]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < MR; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0 && i0 + r < m && j < n)
      dst[(size_t)(i0 + r) * n + j] = s;
  }
}

#define GEMM_Q8(MR)                                                            \
  extern "C" __global__ void gemm_q8_f32_r##MR(                                \
      const float *lhs, const uint8_t *qs, const float *scales, float *dst,    \
      const int m, const int n, const int k) {                                 \
    gemm_q8_rows<MR>(lhs, qs, scales, dst, m, n, k);                           \
  }

GEMM_Q8(1)
GEMM_Q8(2)
GEMM_Q8(4)
GEMM_Q8(8)
GEMM_Q8(16)

// dst[i] = qs[i] * scales[i / 32], the whole weight back to f32 so cuBLAS can
// take over when m is large enough that the weight stream no longer dominates.
extern "C" __global__ void dequant_q8_f32(const uint8_t *qs, const float *scales,
                                          float *dst, const size_t numel) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numel)
    return;
  dst[i] = (float)(int8_t)qs[i] * scales[i >> 5];
}
