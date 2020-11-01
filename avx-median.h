#pragma once

#include <cstdint>
#include <immintrin.h>

void median_Cpp(const float*, float*, size_t);
void median_Step0(const float*, float*, size_t);
void median_Step1(const float*, float*, size_t);
void median_Step2(const float*, float*, size_t);
void median_Step3(const float*, float*, size_t);
void median_Parallel(const float*, float*, size_t);
void median_Parallel_avx2(const float*, float*, size_t);

#ifdef _MSC_VER
#define KEWB_FORCE_INLINE __forceinline
#else
#define KEWB_FORCE_INLINE __attribute__((__always_inline__)) inline
#endif

using rf512 = __m512;
using ri512 = __m512i;
using r512f = __m512;

using m512 = uint32_t;

void dump_reg(const char* const name, r512f value);
#define DUMP_REG(r) dump_reg(#r, r)

KEWB_FORCE_INLINE r512f
    load_from(float const* psrc)
{
    return _mm512_loadu_ps(psrc);
}

KEWB_FORCE_INLINE r512f
    masked_load_from(float const* psrc, r512f fill, m512 mask)
{
    return _mm512_mask_loadu_ps(fill, (__mmask16)mask, psrc);
}

KEWB_FORCE_INLINE __m512
    load_value(float v)
{
    return _mm512_set1_ps(v);
}

KEWB_FORCE_INLINE void
    store_to_address(void* pdst, __m512 r)
{
    _mm512_mask_storeu_ps(pdst, (__mmask16)0xFFFFu, r);
}

KEWB_FORCE_INLINE void
    masked_store_to(void* pdst, __m512 r, m512 mask)
{
    _mm512_mask_storeu_ps(pdst, (__mmask16)mask, r);
}

template<int A, int B, int C, int D, int E, int F, int G, int H,
    int I, int J, int K, int L, int M, int N, int O, int P>
    KEWB_FORCE_INLINE __m512i
    load_values()
{
    return _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
}

KEWB_FORCE_INLINE __m512
    blend(__m512 r0, __m512 r1, uint32_t mask)
{
    return _mm512_mask_blend_ps((__mmask16)mask, r0, r1);
}

KEWB_FORCE_INLINE __m512
    permute(__m512 r, __m512i perm)
{
    return _mm512_permutexvar_ps(perm, r);
}

template<int BIAS, uint32_t MASK>
KEWB_FORCE_INLINE __m512i
    make_shift_permutation()
{
    constexpr int32_t   a = ((BIAS + 0) % 16) | ((MASK & 1u) ? 0x10 : 0);
    constexpr int32_t   b = ((BIAS + 1) % 16) | ((MASK & 1u << 1u) ? 0x10 : 0);
    constexpr int32_t   c = ((BIAS + 2) % 16) | ((MASK & 1u << 2u) ? 0x10 : 0);
    constexpr int32_t   d = ((BIAS + 3) % 16) | ((MASK & 1u << 3u) ? 0x10 : 0);
    constexpr int32_t   e = ((BIAS + 4) % 16) | ((MASK & 1u << 4u) ? 0x10 : 0);
    constexpr int32_t   f = ((BIAS + 5) % 16) | ((MASK & 1u << 5u) ? 0x10 : 0);
    constexpr int32_t   g = ((BIAS + 6) % 16) | ((MASK & 1u << 6u) ? 0x10 : 0);
    constexpr int32_t   h = ((BIAS + 7) % 16) | ((MASK & 1u << 7u) ? 0x10 : 0);
    constexpr int32_t   i = ((BIAS + 8) % 16) | ((MASK & 1u << 8u) ? 0x10 : 0);
    constexpr int32_t   j = ((BIAS + 9) % 16) | ((MASK & 1u << 9u) ? 0x10 : 0);
    constexpr int32_t   k = ((BIAS + 10) % 16) | ((MASK & 1u << 10u) ? 0x10 : 0);
    constexpr int32_t   l = ((BIAS + 11) % 16) | ((MASK & 1u << 11u) ? 0x10 : 0);
    constexpr int32_t   m = ((BIAS + 12) % 16) | ((MASK & 1u << 12u) ? 0x10 : 0);
    constexpr int32_t   n = ((BIAS + 13) % 16) | ((MASK & 1u << 13u) ? 0x10 : 0);
    constexpr int32_t   o = ((BIAS + 14) % 16) | ((MASK & 1u << 14u) ? 0x10 : 0);
    constexpr int32_t   p = ((BIAS + 15) % 16) | ((MASK & 1u << 15u) ? 0x10 : 0);

    return _mm512_setr_epi32(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

template<int R>
KEWB_FORCE_INLINE __m512
    rotate(__m512 r0)
{
    if constexpr ((R % 16) == 0)
    {
        return r0;
    }
    else
    {
        constexpr int    S = (R > 0) ? (16 - (R % 16)) : -R;
        constexpr int    A = (S + 0) % 16;
        constexpr int    B = (S + 1) % 16;
        constexpr int    C = (S + 2) % 16;
        constexpr int    D = (S + 3) % 16;
        constexpr int    E = (S + 4) % 16;
        constexpr int    F = (S + 5) % 16;
        constexpr int    G = (S + 6) % 16;
        constexpr int    H = (S + 7) % 16;
        constexpr int    I = (S + 8) % 16;
        constexpr int    J = (S + 9) % 16;
        constexpr int    K = (S + 10) % 16;
        constexpr int    L = (S + 11) % 16;
        constexpr int    M = (S + 12) % 16;
        constexpr int    N = (S + 13) % 16;
        constexpr int    O = (S + 14) % 16;
        constexpr int    P = (S + 15) % 16;

        return _mm512_permutexvar_ps(_mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P), r0);
    }
}

template<int R>
KEWB_FORCE_INLINE __m512
    rotate_up(__m512 r0)
{
    static_assert(R >= 0);
    return rotate<R>(r0);
}

template<int S>
KEWB_FORCE_INLINE void
    in_place_shift_down_with_carry(__m512& lo, __m512& hi)
{
    static_assert(S >= 0 && S <= 16);

    constexpr uint32_t  zmask = (0xFFFFu >> (unsigned)S);
    constexpr uint32_t  bmask = ~zmask & 0xFFFFu;
    __m512i             perm = make_shift_permutation<S, bmask>();

    lo = _mm512_permutex2var_ps(lo, perm, hi);
    hi = _mm512_maskz_permutex2var_ps((__mmask16)zmask, hi, perm, hi);
}


template<int S>
KEWB_FORCE_INLINE constexpr uint32_t
    shift_up_blend_mask()
{
    static_assert(S >= 0 && S <= 16);
    return (0xFFFFu << (unsigned)S) & 0xFFFFu;
}

template<int S>
KEWB_FORCE_INLINE __m512
    shift_up_with_carry(__m512 lo, __m512 hi)
{
    return blend(rotate_up<S>(lo), rotate_up<S>(hi), shift_up_blend_mask<S>());
}

KEWB_FORCE_INLINE __m512
    mask_permute(__m512 r0, __m512 r1, __m512i perm, uint32_t mask)
{
    return _mm512_mask_permutexvar_ps(r0, (__mmask16)mask, perm, r1);
}

KEWB_FORCE_INLINE __m512
    minimum(__m512 r0, __m512 r1)
{
    return _mm512_min_ps(r0, r1);
}

__m512
    KEWB_FORCE_INLINE maximum(__m512 r0, __m512 r1)
{
    return _mm512_max_ps(r0, r1);
}

KEWB_FORCE_INLINE __m512
mask_minimum(__m512 old, __m512 r0, __m512 r1, m512 mask)
{
    return _mm512_mask_min_ps(old, mask, r0, r1);
}

KEWB_FORCE_INLINE __m512
mask_maximum(__m512 old, __m512 r0, __m512 r1, m512 mask)
{
    return _mm512_mask_max_ps(old, mask, r0, r1);
}

KEWB_FORCE_INLINE __m512
    compare_with_exchange(__m512 vals, __m512i perm, m512 mask)
{
    __m512  exch = permute(vals, perm);
    __m512  vmin = minimum(vals, exch);
    __m512  vmax = maximum(vals, exch);

    return blend(vmin, vmax, mask);
}

template<unsigned A = 0, unsigned B = 0, unsigned C = 0, unsigned D = 0,
    unsigned E = 0, unsigned F = 0, unsigned G = 0, unsigned H = 0,
    unsigned I = 0, unsigned J = 0, unsigned K = 0, unsigned L = 0,
    unsigned M = 0, unsigned N = 0, unsigned O = 0, unsigned P = 0>
    KEWB_FORCE_INLINE constexpr uint32_t
    make_bitmask()
{
    static_assert((A < 2) && (B < 2) && (C < 2) && (D < 2) &&
        (E < 2) && (F < 2) && (G < 2) && (H < 2) &&
        (I < 2) && (J < 2) && (K < 2) && (L < 2) &&
        (M < 2) && (N < 2) && (O < 2) && (P < 2));

    return ((A << 0) | (B << 1) | (C << 2) | (D << 3) |
        (E << 4) | (F << 5) | (G << 6) | (H << 7) |
        (I << 8) | (J << 9) | (K << 10) | (L << 11) |
        (M << 12) | (N << 13) | (O << 14) | (P << 15));
}

template<unsigned... IDXS>
KEWB_FORCE_INLINE auto
    make_permute()
{
    static_assert(sizeof...(IDXS) == 8 || sizeof...(IDXS) == 16);

    if constexpr (sizeof...(IDXS) == 8)
    {
        return _mm256_setr_epi32(IDXS...);
    }
    else
    {
        return load_values<IDXS...>();
    }
}

KEWB_FORCE_INLINE __m512
    sort_two_lanes_of_7(rf512 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                   0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
    //                                  ---------------------------------------------------
    ri512 const     perm0 = make_permute<4, 5, 6, 3, 0, 1, 2, 7, 12, 13, 14, 11, 8, 9, 10, 15>();
    constexpr m512  mask0 = make_bitmask<0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0>();

    ri512 const     perm1 = make_permute<2, 3, 0, 1, 6, 5, 4, 7, 10, 11, 8, 9, 14, 13, 12, 15>();
    constexpr m512  mask1 = make_bitmask<0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0>();

    ri512 const     perm2 = make_permute<1, 0, 4, 5, 2, 3, 6, 7, 9, 8, 12, 13, 10, 11, 14, 15>();
    constexpr m512  mask2 = make_bitmask<0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0>();

    ri512 const     perm3 = make_permute<0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 11, 10, 13, 12, 14, 15>();
    constexpr m512  mask3 = make_bitmask<0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0>();

    ri512 const     perm4 = make_permute<0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15>();
    constexpr m512  mask4 = make_bitmask<0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0>();

    ri512 const     perm5 = make_permute<0, 2, 1, 4, 3, 6, 5, 7, 8, 10, 9, 12, 11, 14, 13, 15>();
    constexpr m512  mask5 = make_bitmask<0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}
