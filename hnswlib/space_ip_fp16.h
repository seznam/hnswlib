#pragma once

#include "hnswlib.h"

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

namespace hnswlib {


    // AVX version, 16 floats per operation
    static float InnerProductSIMD16Ext_FP16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        __m128i *pVect1 = (__m128i *) pVect1v;
        __m128i *pVect2 = (__m128i *) pVect2v;
        size_t iterations = *((size_t *) qty_ptr) / 16;

        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();

        for (size_t i = 0; i < iterations; ++i) {
            sum1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(pVect1)), _mm256_cvtph_ps(_mm_loadu_si128(pVect2)), sum1);
            pVect1 += 1;
            pVect2 += 1;

            sum2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(pVect1)), _mm256_cvtph_ps(_mm_loadu_si128(pVect2)), sum2);
            pVect1 += 1;
            pVect2 += 1;
        }

        return 1.0f - horizontalSum(_mm256_add_ps(sum1, sum2));
    }


    class InnerProductSpace_FP16 : public hnswlib::SpaceInterface<float> {
        hnswlib::DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpace_FP16(size_t dim) {
            if (dim % 16 == 0) {
                fstdistfunc_ = InnerProductSIMD16Ext_FP16;
            } else {
                throw std::runtime_error("Data type not supported!");
            }

            dim_ = dim;
            data_size_ = dim * sizeof(uint16_t);
        }

        ~InnerProductSpace_FP16() {}

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
