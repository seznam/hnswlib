#pragma once

#include "hnswlib.h"

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

namespace hnswlib {

    // x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
    inline float horizontalSum(const __m256 x)
    {
        // hiQuad = ( x7, x6, x5, x4 )
        const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
        // loQuad = ( x3, x2, x1, x0 )
        const __m128 loQuad = _mm256_castps256_ps128(x);
        // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
        const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
        // hiDual = ( -, -, x3 + x7, x2 + x6 )
        const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
        // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
        const __m128 sumDual = _mm_add_ps(sumQuad, hiDual);
        // hi = ( -, -, -, x1 + x3 + x5 + x7 )
        const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
        // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
        const __m128 sum = _mm_add_ss(sumDual, hi);
        return _mm_cvtss_f32(sum);
    }

    // AVX version, 16 floats per operation
    static float L2SqrSIMD16Ext_FP16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        __m128i *pVect1 = (__m128i *) pVect1v;
        __m128i *pVect2 = (__m128i *) pVect2v;
        size_t iterations = *((size_t *) qty_ptr) / 16;

        __m256 diff1, diff2;
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();

        for (size_t i = 0; i < iterations; ++i) {
            diff1 = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(pVect1)), _mm256_cvtph_ps(_mm_loadu_si128(pVect2)));
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            pVect1 += 1;
            pVect2 += 1;

            diff2 = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(pVect1)), _mm256_cvtph_ps(_mm_loadu_si128(pVect2)));
            sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
            pVect1 += 1;
            pVect2 += 1;
        }

        return horizontalSum(_mm256_add_ps(sum1, sum2));
    }


    class L2Space_FP16 : public hnswlib::SpaceInterface<float> {
        hnswlib::DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space_FP16(size_t dim) {
            if (dim % 16 == 0) {
                fstdistfunc_ = L2SqrSIMD16Ext_FP16;
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
