// GEMV kernels adapted from MLX:
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/gemv.metal
// Copyright © 2023-2024 Apple Inc.
//
// Compared to upstream, this copy is trimmed to what the xn backend
// dispatches: the axpby (bias fusion), gather and multi-dim batch variants
// are removed (batching is always a single stride here), and the accumulators
// are `float` instead of `T` so that f16/bf16 matmuls keep the f32
// accumulation semantics used by every other kernel in this backend.
//
// `gemv`   computes out = mat * vec with mat (out_size, in_size), row-major
//          with leading dimension `mat_ld` — the matmul_t/decode layout.
// `gemv_t` computes out = mat^T * vec with mat (in_size, out_size) — the
//          plain matmul layout.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
/// Matrix vector multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
struct GEMVKernel {
  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  static_assert(
      SN == 8 || SN == 16 || SN == 32,
      "gemv block must have a width of 8, 16, or 32");

  // - The matrix of size (M = out_vec_size, K = in_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM rows
  //    and the corresponding scalar from the vector
  // 2. The thread then multiplies and adds to accumulate its local result for
  //    the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated blockM outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN*(blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  static METAL_FUNC void
  load_unsafe(const device T* src, thread T dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = src[src_offset + tn];
    }
  }

  static METAL_FUNC void load_safe(
      const device T* src,
      thread T dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src[src_offset + tn];
      }
    } else { // Edgecase
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size ? src[src_offset + tn] : 0;
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      device T* out_vec,
      const constant int& in_vec_size,
      const constant int& out_vec_size,
      const constant int& matrix_ld,
      threadgroup float* tgp_memory,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    thread float result[TM] = {0};
    thread T inter[TN];
    thread T v_coeff[TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position
    int out_row = tid.x * blockM + bm;

    // Exit simdgroup if rows out of bound
    if (out_row >= out_vec_size)
      return;

    // Adjust tail simdgroup to ensure in bound reads
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    // Advance matrix
    mat += out_row * matrix_ld;

    constexpr const uniform<int> loop_stride = make_uniform(blockN);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Loop over in_vec in blocks of blockN
    for (int i = 0; i < n_iter; ++i) {
      load_unsafe(in_vec, v_coeff, bn);

      // Per thread work loop
      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_unsafe(mat, inter, mat_offset + bn);

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += float(inter[tn]) * float(v_coeff[tn]);
        }

        mat_offset += matrix_ld;
      }

      bn += blockN;
    }

    if (leftover > 0) {
      load_safe(in_vec, v_coeff, bn, in_size);

      // Per thread work loop
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        // Load for the row
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);

        // Accumulate results
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += float(inter[tn]) * float(v_coeff[tn]);
        }
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[tm];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    // Write outputs
    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        out_vec[out_row + tm] = static_cast<T>(result[tm]);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector matrix multiplication
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
struct GEMVTKernel {
  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup can only have 32 threads");

  // - The matrix of size (M = in_vec_size, N = out_vec_size) is divided up
  //   into blocks of (blockM, blockN) divided among threadgroups
  // - Every thread works on a block of (TM, TN)
  // - We assume each threadgroup has (threadsN, threadsM, 1) threads
  //
  // 1. A thread loads TN elements each from mat along TM contiguous rows
  //    and the corresponding scalar from the vector
  // 2. The thread then accumulates its local result for the block
  // 3. At the end, each thread has accumulated results over all blocks across
  //    the rows. These are then summed up across the threadgroup
  // 4. Each threadgroup writes its accumulated BN * TN outputs
  //
  // Edge case handling:
  // - The threadgroup with the largest tid has blocks that exceed the matrix
  //   * The blocks that start outside the matrix are never read (thread results
  //     remain zero)
  //   * The last thread that partially overlaps with the matrix is shifted
  //     inwards such that the thread block fits exactly in the matrix

  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? BM*(blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      device T* out_vec,
      const constant int& in_vec_size,
      const constant int& out_vec_size,
      const constant int& matrix_ld,
      threadgroup float* tgp_memory,
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    // Appease compiler
    (void)lid;

    // Thread local accumulation results
    float result[TN] = {0};
    T inter[TN];
    T v_coeff[TM];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgM = BN != 1 ? (simd_gid / BN) : int(simd_gid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = SM * sgM;
    const int simdN = SN * sgN;

    int cm = (simdM + thrM);
    int cn = (simdN + thrN);

    int bm = cm * TM;
    int bn = cn * TN;

    int out_col = tid.x * blockN + bn;

    constexpr const uniform<int> loop_stride = make_uniform(blockM);
    const uniform<int> in_size = make_uniform(in_vec_size);
    const uniform<int> n_iter = in_size / loop_stride;
    const uniform<int> last_iter = loop_stride * n_iter;
    const uniform<int> leftover = in_size - last_iter;

    // Edgecase handling
    if (out_col < out_vec_size) {
      out_col = out_col + TN < out_vec_size ? out_col : out_vec_size - TN;

      // Per thread accumulation main loop
      for (int i = 0; i < n_iter; ++i) {
        // Adding a threadgroup_barrier improves performance slightly
        // This is possibly it may help exploit cache better
        threadgroup_barrier(mem_flags::mem_none);

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          v_coeff[tm] = in_vec[bm + tm];
        }

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * matrix_ld + out_col + tn];
          }
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += float(v_coeff[tm]) * float(inter[tn]);
          }
        }

        bm += blockM;
      }

      if (leftover > 0) {
        for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
          v_coeff[tm] = in_vec[bm + tm];

          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * matrix_ld + out_col + tn];
          }

          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += float(v_coeff[tm]) * float(inter[tn]);
          }
        }
      }
    }

    // Simdgroup accumulations
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }

    // Threadgroup accumulation results
    if (needs_tgp_reduction) {
      threadgroup float* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          tgp_results[tn] = result[tn];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgm = 1; sgm < BM; sgm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += tgp_results[sgm * (blockN + TN) + tn];
            }
          }
        }
      }
    }

    // Threadgroup accumulation and writing out results
    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        out_vec[out_col + j] = static_cast<T>(result[j]);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Kernel entry points
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM* BN * 32)]] void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& matrix_ld [[buffer(5)]],
    const constant size_t& vector_batch_stride [[buffer(6)]],
    const constant size_t& matrix_batch_stride [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVKernel<T, BM, BN, SM, SN, TM, TN>;
  threadgroup float tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  in_vec += tid.z * vector_batch_stride;
  mat += tid.z * matrix_batch_stride;
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      out_vec,
      in_vec_size,
      out_vec_size,
      matrix_ld,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

template <
    typename T,
    const int BM, /* Threadgroup rows (in simdgroups) */
    const int BN, /* Threadgroup cols (in simdgroups) */
    const int SM, /* Simdgroup rows (in threads) */
    const int SN, /* Simdgroup cols (in threads) */
    const int TM, /* Thread rows (in elements) */
    const int TN> /* Thread cols (in elements) */
[[kernel, max_total_threads_per_threadgroup(BM* BN * 32)]] void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    device T* out_vec [[buffer(2)]],
    const constant int& in_vec_size [[buffer(3)]],
    const constant int& out_vec_size [[buffer(4)]],
    const constant int& matrix_ld [[buffer(5)]],
    const constant size_t& vector_batch_stride [[buffer(6)]],
    const constant size_t& matrix_batch_stride [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using gemv_kernel = GEMVTKernel<T, BM, BN, SM, SN, TM, TN>;
  threadgroup float tgp_memory
      [gemv_kernel::tgp_mem_size == 0 ? 1 : gemv_kernel::tgp_mem_size];

  in_vec += tid.z * vector_batch_stride;
  mat += tid.z * matrix_batch_stride;
  out_vec += tid.z * out_vec_size;

  gemv_kernel::run(
      mat,
      in_vec,
      out_vec,
      in_vec_size,
      out_vec_size,
      matrix_ld,
      gemv_kernel::tgp_mem_size == 0 ? nullptr : tgp_memory,
      tid,
      lid,
      simd_gid,
      simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// Instantiations (only the block shapes the host-side heuristic selects)
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemv_helper(kname, name, itype, bm, bn, sm, sn, tm, tn)  \
  template [[host_name(#kname "_" #name "_bm" #bm "_bn" #bn "_sm" #sm        \
                       "_sn" #sn "_tm" #tm "_tn" #tn)]] [[kernel]] void      \
  kname<itype, bm, bn, sm, sn, tm, tn>(                                      \
      const device itype* mat [[buffer(0)]],                                 \
      const device itype* in_vec [[buffer(1)]],                              \
      device itype* out_vec [[buffer(2)]],                                   \
      const constant int& in_vec_size [[buffer(3)]],                         \
      const constant int& out_vec_size [[buffer(4)]],                        \
      const constant int& matrix_ld [[buffer(5)]],                           \
      const constant size_t& vector_batch_stride [[buffer(6)]],              \
      const constant size_t& matrix_batch_stride [[buffer(7)]],              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_gemv_blocks(name, itype)                        \
  instantiate_gemv_helper(gemv, name, itype, 4, 1, 1, 32, 1, 4)     \
  instantiate_gemv_helper(gemv, name, itype, 4, 1, 1, 32, 4, 4)     \
  instantiate_gemv_helper(gemv, name, itype, 8, 1, 1, 32, 4, 4)     \
  instantiate_gemv_helper(gemv_t, name, itype, 1, 2,  8, 4, 4, 1)   \
  instantiate_gemv_helper(gemv_t, name, itype, 1, 2,  8, 4, 4, 4)   \
  instantiate_gemv_helper(gemv_t, name, itype, 1, 4,  8, 4, 4, 4)   \
  instantiate_gemv_helper(gemv_t, name, itype, 1, 16, 8, 4, 4, 4)   \
  instantiate_gemv_helper(gemv_t, name, itype, 1, 16, 4, 8, 4, 4) // clang-format on

instantiate_gemv_blocks(f32, float);
instantiate_gemv_blocks(f16, half);
#if defined(__HAVE_BFLOAT__)
instantiate_gemv_blocks(bf16, bfloat);
#endif
