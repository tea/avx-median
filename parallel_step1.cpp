#include "avx-median.h"

template<int Offset>
struct StepwiseGather
{
    static_assert(Offset < 16, "");

    template<int V> static constexpr uint8_t C = V < 0 ? 0 : V > 15 ? 0 : (uint8_t)V;

    template<int RegStart>
    static __m512i stepwise_permute()
    {
        return make_permute<C<Offset - RegStart>, C<Offset + 2 - RegStart>, C<Offset + 4 - RegStart>, C<Offset + 6 - RegStart>,
            C<Offset + 8 - RegStart>, C<Offset + 10 - RegStart>, C<Offset + 12 - RegStart>, C<Offset + 14 - RegStart>,
            C<Offset + 16 - RegStart>, C<Offset + 18 - RegStart>, C<Offset + 20 - RegStart>, C<Offset + 22 - RegStart>,
            C<Offset + 24 - RegStart>, C<Offset + 26 - RegStart>, C<Offset + 28 - RegStart>, C<Offset + 30 - RegStart>>();
    }

    template<unsigned int RegStart>
    static constexpr m512 make_loadmask()
    {
        static_assert(Offset < RegStart, "");
        constexpr auto loaded = (RegStart - Offset + 1) / 2;
        if constexpr (loaded >= 16)
            return 0;
        else
            return (0xFFFFul << loaded) & 0xFFFF;
    }

    KEWB_FORCE_INLINE
    __m512 operator()(__m512 lo, __m512 med, __m512 hi) const noexcept
    {
        __m512 data = permute(lo, perm_lo);
        data = mask_permute(data, med, perm_med, mask_med);
        if constexpr (mask_hi != 0)
            data = mask_permute(data, hi, perm_hi, mask_hi);
        return data;
    }

    const __m512i perm_lo = stepwise_permute<0>();
    const __m512i perm_med = stepwise_permute<16>();
    const __m512i perm_hi = stepwise_permute<32>();

    static constexpr auto mask_med = make_loadmask<16>();
    static constexpr auto mask_hi = make_loadmask<32>();
};

static const     auto Ys_perm_lo = make_permute<0, 7, 2, 9, 4, 11, 6, 13, 8, 15, 10, 0, 12, 0, 14, 0>();
static constexpr auto Ys_mask_hi = make_bitmask<0, 0, 0, 0, 0,  0, 0,  0, 0,  0,  0, 1,  0, 1,  0, 1>();
static const     auto Ys_perm_hi = make_permute<0, 0, 0, 0, 0,  0, 0,  0, 0,  0,  0, 1,  0, 3,  0, 5>();

static const     auto pairwise_broadcast_perm_lo = make_permute<0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>();
static const     auto pairwise_broadcast_perm_hi = make_permute<8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15>();

static const StepwiseGather<1> G1;
static const StepwiseGather<2> G2;
static const StepwiseGather<3> G3;
static const StepwiseGather<4> G4;
static const StepwiseGather<5> G5;
static const StepwiseGather<6> G6;

KEWB_FORCE_INLINE
static void sort(rf512& l, rf512& r)
{
    rf512 tmp = minimum(l, r);
    r = maximum(l, r);
    l = tmp;
}

KEWB_FORCE_INLINE
static void process32(rf512& lo, rf512 med, rf512& hi)
{
    rf512 Ys_lo = permute(lo, Ys_perm_lo);
    Ys_lo = mask_permute(Ys_lo, med, Ys_perm_hi, Ys_mask_hi);
    rf512 Ys_hi = permute(med, Ys_perm_lo);
    Ys_hi = mask_permute(Ys_hi, hi, Ys_perm_hi, Ys_mask_hi);

    rf512 s1 = G1(lo, med, hi);
    rf512 s2 = G2(lo, med, hi);
    rf512 s3 = G3(lo, med, hi);
    rf512 s4 = G4(lo, med, hi);
    rf512 s5 = G5(lo, med, hi);
    rf512 s6 = G6(lo, med, hi);
    sort(s1, s2); sort(s3, s4); sort(s5, s6);
    sort(s1, s3); sort(s2, s5); sort(s4, s6);
    s2 = maximum(s1, s2); sort(s3, s4); s5 = minimum(s5, s6);
    s3 = maximum(s2, s3); s4 = minimum(s4, s5);
    sort(s3, s4);

    rf512 tmp = permute(s3, pairwise_broadcast_perm_lo);
    Ys_lo = maximum(Ys_lo, tmp);
    tmp = permute(s4, pairwise_broadcast_perm_lo);
    lo = minimum(Ys_lo, tmp);

    tmp = permute(s3, pairwise_broadcast_perm_hi);
    Ys_hi = maximum(Ys_hi, tmp);
    tmp = permute(s4, pairwise_broadcast_perm_hi);
    hi = minimum(Ys_hi, tmp);
}

void median_Parallel_step1(const float* psrc, float* pdst, size_t buf_len)
{
    __m512      prev;   //- Bottom of the input data window
    __m512      curr_lo, curr_hi;   //- Middle of the input data window
    __m512      next;   //- Top of the input data window
    __m512      lo, med, hi;
    m512        mask;   //- Trailing boundary mask

    if (buf_len == 0)
        return;
    if (buf_len == 1)
    {
        *pdst = *psrc;
        return;
    }

    rf512 const     first = load_value(psrc[0]);
    rf512 const     last = load_value(psrc[buf_len - 1]);

    if (buf_len < 16)
    {
        prev = first;
        mask = ~(0xffffffff << buf_len);
        curr_lo = masked_load_from(psrc, last, mask);
        curr_hi = next = last;

        lo = shift_up_with_carry<3>(prev, curr_lo);
        med = shift_up_with_carry<3>(curr_lo, curr_hi);
        hi = shift_up_with_carry<3>(curr_hi, next);

        process32(lo, med, hi);
        masked_store_to(pdst, lo, mask);
        return;
    }

    curr_hi = first;
    next = load_from(psrc); psrc += 16; buf_len -= 16;
    
    while(buf_len >= 32)
    {
        prev = curr_hi;
        curr_lo = next;
        curr_hi = load_from(psrc); psrc += 16;
        next = load_from(psrc); psrc += 16;

        lo = shift_up_with_carry<3>(prev, curr_lo);
        med = shift_up_with_carry<3>(curr_lo, curr_hi);
        hi = shift_up_with_carry<3>(curr_hi, next);

        process32(lo, med, hi);

        store_to_address(pdst, lo); pdst += 16;
        store_to_address(pdst, hi); pdst += 16;
        buf_len -= 32;
    }

    prev = curr_hi;
    curr_lo = next;
    if (buf_len >= 16)
    {
        curr_hi = load_from(psrc); psrc += 16; buf_len -= 16;
        mask = ~(0xffffffff << buf_len);
        next = masked_load_from(psrc, last, mask);

        lo = shift_up_with_carry<3>(prev, curr_lo);
        med = shift_up_with_carry<3>(curr_lo, curr_hi);
        hi = shift_up_with_carry<3>(curr_hi, next);

        process32(lo, med, hi);

        store_to_address(pdst, lo); pdst += 16;
        store_to_address(pdst, hi); pdst += 16;

        if (buf_len > 0)
        {
            prev = curr_hi;
            curr_lo = next;
            curr_hi = next = last;

            lo = shift_up_with_carry<3>(prev, curr_lo);
            med = shift_up_with_carry<3>(curr_lo, curr_hi);
            hi = shift_up_with_carry<3>(curr_hi, next);

            process32(lo, med, hi);
            masked_store_to(pdst, lo, mask);
        }
    }
    else
    {
        mask = ~(0xffffffff << buf_len);
        curr_hi = masked_load_from(psrc, last, mask);
        next = last;

        lo = shift_up_with_carry<3>(prev, curr_lo);
        med = shift_up_with_carry<3>(curr_lo, curr_hi);
        hi = shift_up_with_carry<3>(curr_hi, next);

        process32(lo, med, hi);

        store_to_address(pdst, lo); pdst += 16;
        masked_store_to(pdst, hi, mask);
    }
}
