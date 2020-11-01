#include "avx-median.h"

// Adaptation of https://habr.com/ru/post/204682/ algorithm using AVX2

KEWB_FORCE_INLINE __m256
load_value_avx2(float v)
{
    return _mm256_set1_ps(v);
}

KEWB_FORCE_INLINE __m256
load_from_avx2(float const* psrc)
{
    return _mm256_loadu_ps(psrc);
}

KEWB_FORCE_INLINE void
store_to_address(float* pdst, __m256 r)
{
    _mm256_storeu_ps(pdst, r);
}

KEWB_FORCE_INLINE __m256
masked_load_from(float const* psrc, __m256 fill, __m256i mask)
{
    __m256 values = _mm256_maskload_ps(psrc, mask);
    return _mm256_blendv_ps(fill, values, _mm256_castsi256_ps(mask));
}

KEWB_FORCE_INLINE
static __m256 minimum(__m256 l, __m256 r)
{
    return _mm256_min_ps(l, r);
}

KEWB_FORCE_INLINE
static __m256 maximum(__m256 l, __m256 r)
{
    return _mm256_max_ps(l, r);
}

template<int R>
KEWB_FORCE_INLINE __m256
rotate_up(__m256 r0)
{
    static_assert(R >= 0);
    if constexpr ((R % 8) == 0)
    {
        return r0;
    }
    else
    {
        constexpr int    S = 8 - (R % 8);
        constexpr int    A = (S + 0) % 8;
        constexpr int    B = (S + 1) % 8;
        constexpr int    C = (S + 2) % 8;
        constexpr int    D = (S + 3) % 8;
        constexpr int    E = (S + 4) % 8;
        constexpr int    F = (S + 5) % 8;
        constexpr int    G = (S + 6) % 8;
        constexpr int    H = (S + 7) % 8;

        return _mm256_permutevar8x32_ps(r0, _mm256_setr_epi32(A, B, C, D, E, F, G, H));
    }
}

template<int S>
KEWB_FORCE_INLINE __m256
blend(__m256 r0, __m256 r1)
{
    return _mm256_blend_ps(r0, r1, S);
}

template<int S>
KEWB_FORCE_INLINE constexpr uint8_t
shift_up_blend_mask_avx2()
{
    static_assert(S >= 0 && S <= 8);
    return (0xFFu << (unsigned)S) & 0xFFu;
}

template<int S>
KEWB_FORCE_INLINE __m256
shift_up_with_carry(__m256 lo, __m256 hi)
{
    return blend<shift_up_blend_mask_avx2<S>()>(rotate_up<S>(lo), rotate_up<S>(hi));
}

KEWB_FORCE_INLINE
static __m256i make_loadmask(size_t value)
{
    return _mm256_cmpgt_epi32(_mm256_set1_epi32((int)value), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}

KEWB_FORCE_INLINE void
masked_store_to(float* pdst, __m256 r, __m256i mask)
{
    _mm256_maskstore_ps(pdst, mask, r);
}

KEWB_FORCE_INLINE
static void sort(__m256& l, __m256& r)
{
    __m256 tmp = minimum(l, r);
    r = maximum(l, r);
    l = tmp;
}

KEWB_FORCE_INLINE
static __m256 process8(__m256 s1, __m256 hi)
{
    __m256 s2 = shift_up_with_carry<7>(s1, hi);
    __m256 s3 = shift_up_with_carry<6>(s1, hi);
    __m256 s4 = shift_up_with_carry<5>(s1, hi);
    __m256 s5 = shift_up_with_carry<4>(s1, hi);
    __m256 s6 = shift_up_with_carry<3>(s1, hi);
    __m256 s7 = shift_up_with_carry<2>(s1, hi);
    sort(s2, s3); sort(s4, s5); sort(s6, s7);
    sort(s1, s3); sort(s5, s7); sort(s4, s6);
    s3 = minimum(s3, s7); sort(s2, s6); sort(s1, s5);
    s3 = minimum(s3, s6); s4 = maximum(s4, s1);
    s3 = minimum(s3, s5); s4 = maximum(s2, s4);
    s4 = maximum(s3, s4);
    return s4;
}

void median_Parallel_avx2(const float* psrc, float* pdst, size_t buf_len)
{
    __m256      prev;   //- Bottom of the input data window
    __m256      curr;   //- Middle of the input data window
    __m256      next;   //- Top of the input data window
    __m256      lo;     //- Primary work register
    __m256      hi;     //- Upper work data register; feeds values into the top of 'lo'
    __m256i     mask;   //- Trailing boundary mask
    __m256      data;   //- Holds output prior to store operation

    __m256 const     first = load_value_avx2(psrc[0]);
    __m256 const     last = load_value_avx2(psrc[buf_len - 1]);

    //- Preload the initial input data window; note the values in the register representing
    //  data preceding the input array are equal to the first element.
    //

    if (buf_len < 8)
    {
        prev = first;
        mask = make_loadmask(buf_len);
        curr = masked_load_from(psrc, last, mask);
        next = last;

        //- Init the work data register to the correct offset in the input data window.
        //
        lo = shift_up_with_carry<3>(prev, curr);
        hi = shift_up_with_carry<3>(curr, next);

        data = process8(lo, hi);
        masked_store_to(pdst, data, mask);
    }
    else
    {
        size_t  read = 0;
        size_t  used = 0;
        size_t  wrote = 0;

        curr = first;
        next = load_from_avx2(psrc);
        read += 8;
        used += 8;

        while (used < (buf_len + 8))
        {
            prev = curr;
            curr = next;

            if (read <= (buf_len - 8))
            {
                next = load_from_avx2(psrc + read);
                read += 8;
            }
            else
            {
                mask = make_loadmask(buf_len - read);
                next = masked_load_from(psrc + read, last, mask);
                read = buf_len;
            }
            used += 8;

            //- Init the work data register to the correct offset in the input data window.
            //
            lo = shift_up_with_carry<3>(prev, curr);
            hi = shift_up_with_carry<3>(curr, next);

            data = process8(lo, hi);

            if (wrote <= (buf_len - 8))
            {
                store_to_address(pdst + wrote, data);
                wrote += 8;
            }
            else
            {
                mask = make_loadmask(buf_len - wrote);
                masked_store_to(pdst + wrote, data, mask);
                wrote = buf_len;
            }
        }
    }
}
