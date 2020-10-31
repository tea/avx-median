#include "avx-median.h"

static const ri512 xes_perm = make_permute<0, 7, 2, 9, 4, 11, 6, 13, 0, 7, 2, 9, 4, 11, 6, 13>();
static const ri512 load_perm_1 = make_permute<1, 2, 3, 4, 5, 6, 0, 0, 3, 4, 5, 6, 7, 8, 0, 0>();
static const ri512 load_perm_2 = make_permute<5, 6, 7, 8, 9, 10, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0>();
static const ri512 twos_perm = make_permute<2, 2, 10, 10, 2, 2, 10, 10, 2, 2, 10, 10, 2, 2, 10, 10>();
static const ri512 threes_perm = make_permute<3, 3, 11, 11, 3, 3, 11, 11, 3, 3, 11, 11, 3, 3, 11, 11>();
static constexpr m512 iter0_mask = make_bitmask<1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0>();
static constexpr m512 iter1_mask = make_bitmask<0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1>();

static constexpr m512 save = make_bitmask<1, 1, 1, 1, 1, 1, 1, 1>();
static constexpr m512 save_mask[2] = { save << 0, save << 8 };

static ri512 const     perm0 = make_permute<0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 15>();
static constexpr m512  mask0 = make_bitmask<0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0>();

static ri512 const     perm1 = make_permute<2, 4, 0, 5, 1, 3, 6, 7, 10, 12, 8, 13, 9, 11, 14, 15>();
static constexpr m512  mask1 = make_bitmask<0, 0, 1, 0, 1, 1, 0, 0,  0,  0, 1,  0, 1,  1,  0,  0>();

static ri512 const     perm2 = make_permute<1, 0, 3, 2, 5, 4, 6, 7, 9, 8, 11, 10, 13, 12, 14, 15>();
static constexpr m512  mask2 = make_bitmask<0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0>();

static ri512 const     perm3 = make_permute<0, 2, 1, 4, 3, 5, 6, 7, 8, 10, 9, 12, 11, 13, 14, 15>();
static constexpr m512  mask3 = make_bitmask<0, 0, 1, 0, 1, 0, 0, 0, 0,  0, 1,  0,  1,  0,  0,  0>();

static ri512 const     perm4 = make_permute<0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 11, 10, 12, 13, 14, 15>();
static constexpr m512  mask4 = make_bitmask<0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0,  1,  0,  0,  0,  0>();

KEWB_FORCE_INLINE __m512
sort_two_lanes_of_6(rf512 vals)
{
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);

    return vals;
}

KEWB_FORCE_INLINE
static rf512 process16(rf512 lo, rf512 hi)
{
    __m512   presorted;
    __m512    work;
    __m512   data;
    __m512   Xes;
    __m512   tmp;

    for (int i = 0; i < 2; ++i)
    {
        Xes = permute(lo, xes_perm);
        presorted = compare_with_exchange(lo, perm0, mask0);
        work = permute(presorted, load_perm_1);
        work = sort_two_lanes_of_6(work);
        tmp = permute(work, twos_perm);
        Xes = mask_maximum(Xes, Xes, tmp, iter0_mask);
        tmp = permute(work, threes_perm);
        Xes = mask_minimum(Xes, Xes, tmp, iter0_mask);

        work = permute(presorted, load_perm_2);
        work = sort_two_lanes_of_6(work);
        tmp = permute(work, twos_perm);
        Xes = mask_maximum(Xes, Xes, tmp, iter1_mask);
        tmp = permute(work, threes_perm);
        Xes = mask_minimum(Xes, Xes, tmp, iter1_mask);
        data = blend(data, Xes, save_mask[i]);
        in_place_shift_down_with_carry<8>(lo, hi);
    }
    return data;
}

void median_Step2(const float* psrc, float* pdst, size_t buf_len)
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
