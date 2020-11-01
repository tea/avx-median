#include "avx-median.h"

// Adaptation of https://habr.com/ru/post/204682/ algorithm
KEWB_FORCE_INLINE
static void sort(rf512& l, rf512& r)
{
    rf512 tmp = minimum(l, r);
    r = maximum(l, r);
    l = tmp;
}

KEWB_FORCE_INLINE
static rf512 process16(rf512 s1, rf512 hi)
{
    rf512 s2 = shift_up_with_carry<15>(s1, hi);
    rf512 s3 = shift_up_with_carry<14>(s1, hi);
    rf512 s4 = shift_up_with_carry<13>(s1, hi);
    rf512 s5 = shift_up_with_carry<12>(s1, hi);
    rf512 s6 = shift_up_with_carry<11>(s1, hi);
    rf512 s7 = shift_up_with_carry<10>(s1, hi);
    sort(s2, s3); sort(s4, s5); sort(s6, s7);
    sort(s1, s3); sort(s5, s7); sort(s4, s6);
    s3 = minimum(s3, s7); sort(s2, s6); sort(s1, s5);
    s3 = minimum(s3, s6); s4 = maximum(s4, s1);
    s3 = minimum(s3, s5); s4 = maximum(s2, s4);
    s4 = maximum(s3, s4);
    return s4;
}

void median_Parallel(const float* psrc, float* pdst, size_t buf_len)
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
