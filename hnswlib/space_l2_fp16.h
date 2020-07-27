#pragma once

#include "hnswlib.h"

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

namespace hnswlib {

    // AVX version, 8 floats per operation
    static float L2SqrSIMD8Ext_FP16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        __m128i *pVect1 = (__m128i *) pVect1v;
        __m128i *pVect2 = (__m128i *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];

        const __m128i *pEnd1 = pVect1 + qty/8;

        __m256 diff, v1, v2;
        __m128i v1i;
        __m128i v2i;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1i = _mm_loadu_si128(pVect1);
            v1 = _mm256_cvtph_ps(v1i);
            pVect1 += 1;

            v2i = _mm_loadu_si128(pVect2);
            v2 = _mm256_cvtph_ps(v2i);
            pVect2 += 1;

            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return (res);
    }

    // 4 floats per operation
    static float L2SqrSIMD4Ext_FP16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[4];
        uint16_t *pVect1 = (uint16_t *) pVect1v;
        uint16_t *pVect2 = (uint16_t *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        const uint16_t *pEnd1 = pVect1 + qty;

        __m128 diff, v1, v2;
        __m128i v1i, v2i;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //v1i = _mm_loadu_si64(pVect1); // FIXME: works on gcc-9 and later
            v1i = _mm_loadl_epi64((__m128i *)pVect1);
            v1 = _mm_cvtph_ps(v1i);
            pVect1 += 4;

            //v2i = _mm_loadu_si64(pVect2); // FIXME: works on gcc-9 and later
            v2i = _mm_loadl_epi64((__m128i *)pVect2);
            v2 = _mm_cvtph_ps(v2i);
            pVect2 += 4;

            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return (res);
    }

    class L2Space_FP16 : public hnswlib::SpaceInterface<float> {
        hnswlib::DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space_FP16(size_t dim) {
            if (dim % 8 == 0) {
                fstdistfunc_ = L2SqrSIMD8Ext_FP16;
            } else if (dim % 4 == 0) {
                fstdistfunc_ = L2SqrSIMD4Ext_FP16;
            } else {
                throw std::runtime_error("Data type not supported!");
            }

            dim_ = dim;
            data_size_ = dim * sizeof(uint16_t);
        }

        ~L2Space_FP16() {}

        virtual size_t get_data_size() override {
            return data_size_;
        }

        virtual hnswlib::DISTFUNC<float> get_dist_func() override {
            return fstdistfunc_;
        }

        virtual void *get_dist_func_param() override {
            return &dim_;
        }
    };

}
