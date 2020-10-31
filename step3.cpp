#include "avx-median.h"

static const ri512 xes_perm = make_permute<0, 7, 2, 9, 4, 11, 6, 13, 0, 7, 2, 9, 4, 11, 6, 13>();
static const ri512 ys_perm = make_permute<1, 2, 0, 0, 7, 8, 0, 0, 5, 6, 0, 0, 11, 12, 0, 0>();

static ri512 const     step1_perm0 = make_permute<0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 15>();
static constexpr m512  step1_mask0 = make_bitmask<0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0>();

static ri512 const     step1_perm1 = make_permute<0, 1, 2, 5, 6, 3, 4, 9, 10, 7, 8, 11, 12, 13, 14, 15>();
static constexpr m512  step1_mask1 = make_bitmask<0, 0, 0, 0, 0, 1, 1, 0,  0, 1, 1,  0,  0,  0,  0, 0>();

static ri512 const    Y_max_perm = make_permute<3, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0>();
static constexpr m512 Y_max_mask = make_bitmask<1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0>();
static ri512 const    Y_min_perm = make_permute<0, 6, 0, 0, 0, 6, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0>();
static constexpr m512 Y_min_mask = make_bitmask<0, 1, 0, 0, 0, 1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0>();

static ri512 const    YS_mix_perm = make_permute<0, 0, 4, 5, 0, 0, 4, 5, 0, 0, 8, 9, 0, 0, 8, 9>();
static constexpr m512 YS_mix_mask = make_bitmask<0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1>();

static ri512 const     step2_perm0 = make_permute<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>();
static constexpr m512  step2_mask0 = make_bitmask<0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1>();
static ri512 const     step2_perm1 = make_permute<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>();
static constexpr m512  step2_mask1 = make_bitmask<0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>();
static ri512 const     step2_perm2 = make_permute<0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15>();
static constexpr m512  step2_mask2 = make_bitmask<0, 0, 1, 0, 0, 0, 1, 0, 0,  0, 1,  0,  0,  0,  1,  0>();

static const ri512 twos_perm = make_permute<1, 1, 5, 5, 9, 9, 13, 13, 1, 1, 5, 5, 9, 9, 13, 13>();
static const ri512 threes_perm = make_permute<2, 2, 6, 6, 10, 10, 14, 14, 2, 2, 6, 6, 10, 10, 14, 14>();

static constexpr m512 save = make_bitmask<1, 1, 1, 1, 1, 1, 1, 1>();
static constexpr m512 save_mask[2] = { save << 0, save << 8 };

KEWB_FORCE_INLINE
static rf512 process16(rf512 lo, rf512 hi)
{
    __m512   work;
    __m512   data;
    __m512   Xes;
    __m512   Ys;
    __m512   tmp;

    for (int i = 0; i < 2; ++i)
    {
        Xes = permute(lo, xes_perm);
        work = compare_with_exchange(lo, step1_perm0, step1_mask0);
        Ys = permute(work, ys_perm);
        work = compare_with_exchange(work, step1_perm1, step1_mask1);

        tmp = permute(work, Y_max_perm);
        Ys = mask_maximum(Ys, Ys, tmp, Y_max_mask);
        tmp = permute(work, Y_min_perm);
        Ys = mask_minimum(Ys, Ys, tmp, Y_min_mask);
        work = mask_permute(Ys, work, YS_mix_perm, YS_mix_mask);

        work = compare_with_exchange(work, step2_perm0, step2_mask0);
        work = compare_with_exchange(work, step2_perm1, step2_mask1);
        work = compare_with_exchange(work, step2_perm2, step2_mask2);

        tmp = permute(work, twos_perm);
        Xes = maximum(Xes, tmp);
        tmp = permute(work, threes_perm);
        Xes = minimum(Xes, tmp);

        data = blend(data, Xes, save_mask[i]);
        in_place_shift_down_with_carry<8>(lo, hi);
    }
    return data;
}

void median_Step3(const float* psrc, float* pdst, size_t buf_len)
{
    __m512      prev;   //- Bottom of the input data window
    __m512      curr;   //- Middle of the input data window
    __m512      next;   //- Top of the input data window
    __m512      lo;     //- Primary work register
    __m512      hi;     //- Upper work data register; feeds values into the top of 'lo'
    m512        mask;   //- Trailing boundary mask
    __m512      data;   //- Holds output prior to store operation

    rf512 const     first = load_value(psrc[0]);
    rf512 const     last = load_value(psrc[buf_len - 1]);

    //- Preload the initial input data window; note the values in the register representing
    //  data preceding the input array are equal to the first element.
    //

    if (buf_len < 16)
    {
        prev = first;
        mask = ~(0xffffffff << buf_len);
        curr = masked_load_from(psrc, last, mask);
        next = last;

        //- Init the work data register to the correct offset in the input data window.
        //
        lo = shift_up_with_carry<3>(prev, curr);
        hi = shift_up_with_carry<3>(curr, next);

        data = process16(lo, hi);
        masked_store_to(pdst, data, mask);
    }
    else
    {
        size_t  read = 0;
        size_t  used = 0;
        size_t  wrote = 0;

        curr = first;
        next = load_from(psrc);
        read += 16;
        used += 16;

        while (used < (buf_len + 16))
        {
            prev = curr;
            curr = next;

            if (read <= (buf_len - 16))
            {
                next = load_from(psrc + read);
                read += 16;
            }
            else
            {
                mask = ~(0xffffffff << (buf_len - read));
                next = masked_load_from(psrc + read, last, mask);
                read = buf_len;
            }
            used += 16;

            //- Init the work data register to the correct offset in the input data window.
            //
            lo = shift_up_with_carry<3>(prev, curr);
            hi = shift_up_with_carry<3>(curr, next);

            data = process16(lo, hi);

            if (wrote <= (buf_len - 16))
            {
                store_to_address(pdst + wrote, data);
                wrote += 16;
            }
            else
            {
                mask = ~(0xffffffff << (buf_len - wrote));
                masked_store_to(pdst + wrote, data, mask);
                wrote = buf_len;
            }
        }
    }
}
