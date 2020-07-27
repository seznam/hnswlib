#pragma once

#include "hnswalg.h"

namespace hnswlib
{

    template<typename dist_t>
    class HierarchicalNSW_FP16 : public hnswlib::AlgorithmInterface<dist_t> {
    protected:
        hnswlib::HierarchicalNSW<dist_t> hierarchical_nsw_;
        size_t dim_;
        size_t data_size_;

        void initData(hnswlib::SpaceInterface<dist_t> *s) {
            dim_ = *((size_t*)s->get_dist_func_param());
            data_size_ = s->get_data_size();
            //data_.resize(dim_, 0);
        }
    public:
        HierarchicalNSW_FP16(hnswlib::SpaceInterface<dist_t> *s) : hierarchical_nsw_(s) {
            initData(s);
        }

        HierarchicalNSW_FP16(hnswlib::SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements = 0, bool search_only = false)
          : hierarchical_nsw_(s, location, nmslib, max_elements, search_only)
        {
            initData(s);
        }

        HierarchicalNSW_FP16(hnswlib::SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100)
          : hierarchical_nsw_(s, max_elements, M, ef_construction, random_seed)
        {
            initData(s);
        }

        virtual ~HierarchicalNSW_FP16() {
        }

        virtual void addPoint(const void *datapoint, hnswlib::labeltype label) override;

        virtual std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> searchKnn(const void *datapoint, size_t result_count) const override;

        template <typename Comp>
        std::vector<std::pair<dist_t, hnswlib::labeltype>> searchKnn(const void *datapoint, size_t result_count, Comp comparator) {
            uint64_t *halfDatapoint = (uint64_t *) alloca(data_size_);
            convertSpToHp((float *) datapoint, halfDatapoint, dim_);
            return hierarchical_nsw_.searchKnn(halfDatapoint, result_count);
        }

        virtual void saveIndex(const std::string &location) override {
            hierarchical_nsw_.saveIndex(location);
        }

        void setEf(size_t ef) {
            hierarchical_nsw_.setEf(ef);
        }

        static void convertSpToHp(const float * inDatapoint, uint64_t * outDatapoint, const size_t dim);
        static void convertHpToSp(const uint64_t * inDatapoint, float * outDatapoint, const size_t dim);
    };

    template<typename dist_t>
    void HierarchicalNSW_FP16<dist_t>::convertHpToSp(const uint64_t * inDatapoint, float * outDatapoint, const size_t dim) {
        for(size_t i = 0; i != dim / 4; i++) {
            __m128i h = _mm_cvtsi64_si128(inDatapoint[i]);
            __m128 f = _mm_cvtph_ps(h);
            _mm_storeu_ps(outDatapoint, f);
            outDatapoint += 4;
        }
    }

    template<typename dist_t>
    void HierarchicalNSW_FP16<dist_t>::convertSpToHp(const float * inDatapoint, uint64_t * outDatapoint, const size_t dim) {
        for(size_t i = 0; i != dim / 4; i++) {
            __m128 inValue = _mm_loadu_ps(inDatapoint);
            inDatapoint += 4;
            __m128i v_hf = _mm_cvtps_ph(inValue, _MM_FROUND_TO_NEAREST_INT);
            //_mm_storeu_si64(outAddr, v_hf); // FIXME: works on gcc-9 and later
            _mm_storel_epi64((__m128i *) outDatapoint, v_hf);
            outDatapoint += 1;
        }
    }

    template<typename dist_t>
    void HierarchicalNSW_FP16<dist_t>::addPoint(const void *datapoint, hnswlib::labeltype label) {
        uint64_t *halfDatapoint = (uint64_t *) alloca(data_size_);
        convertSpToHp((float *) datapoint, halfDatapoint, dim_);
        hierarchical_nsw_.addPoint(halfDatapoint, label);
    }

    template<typename dist_t>
    std::priority_queue<std::pair<dist_t, hnswlib::labeltype >> HierarchicalNSW_FP16<dist_t>::searchKnn(const void *datapoint, size_t result_count) const {
        uint64_t *halfDatapoint = (uint64_t *) alloca(data_size_);
        convertSpToHp((float *)datapoint, halfDatapoint, dim_);
        return hierarchical_nsw_.searchKnn(halfDatapoint, result_count);
    }

}
