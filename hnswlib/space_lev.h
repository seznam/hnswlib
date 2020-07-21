#pragma once
#include "hnswlib.h"

namespace hnswlib {

    // levenshtein data point
    struct LevDataPoint {
        ssize_t index = -1; // if index < 0 use data instead
        const char* data =  nullptr;
    };

    /**
     * lev_edit_distance:
     * @len1: The length of @string1.
     * @string1: A sequence of bytes of length @len1, may contain NUL characters.
     * @len2: The length of @string2.
     * @string2: A sequence of bytes of length @len2, may contain NUL characters.
     *
     * Computes Levenshtein edit distance of two strings.
     *
     * Returns: The edit distance.
     **/
    static size_t lev_edit_distance(size_t len1, const char *string1,
            size_t len2, const char *string2)
    {
        size_t i;
        size_t *row;  /* we only need to keep one row of costs */
        size_t *end;
        size_t half;

        /* strip common prefix */
        while (len1 > 0 && len2 > 0 && *string1 == *string2) {
            len1--;
            len2--;
            string1++;
            string2++;
        }

        /* strip common suffix */
        while (len1 > 0 && len2 > 0 && string1[len1-1] == string2[len2-1]) {
            len1--;
            len2--;
        }

        /* catch trivial cases */
        if (len1 == 0)
            return len2;
        if (len2 == 0)
            return len1;

        /* make the inner cycle (i.e. string2) the longer one */
        if (len1 > len2) {
            size_t nx = len1;
            const char *sx = string1;
            len1 = len2;
            len2 = nx;
            string1 = string2;
            string2 = sx;
        }
        /* check len1 == 1 separately */
        if (len1 == 1) {
            return len2 - (memchr(string2, *string1, len2) != NULL);
        }
        len1++;
        len2++;
        half = len1 >> 1;

        /* initalize first row */
        row = (size_t*)alloca(len2*sizeof(size_t));
        if (!row)
            return (size_t)(-1);
        end = row + len2 - 1;
        for (i = 0; i < len2 - half; i++)
            row[i] = i;

        /* in this case we don't have to scan two corner triangles (of size len1/2)
         * in the matrix because no best path can go throught them. note this
         * breaks when len1 == len2 == 2 so the memchr() special case above is
         * necessary */
        row[0] = len1 - half - 1;
        for (i = 1; i < len1; i++) {
            size_t *p;
            const char char1 = string1[i - 1];
            const char *char2p;
            size_t D, x;
            /* skip the upper triangle */
            if (i >= len1 - half) {
                size_t offset = i - (len1 - half);
                size_t c3;

                char2p = string2 + offset;
                p = row + offset;
                c3 = *(p++) + (char1 != *(char2p++));
                x = *p;
                x++;
                D = x;
                if (x > c3)
                    x = c3;
                *(p++) = x;
            }
            else {
                p = row + 1;
                char2p = string2;
                D = x = i;
            }
            /* skip the lower triangle */
            if (i <= half + 1)
                end = row + len2 + i - half - 2;
            /* main */
            while (p <= end) {
                size_t c3 = --D + (char1 != *(char2p++));
                x++;
                if (x > c3)
                    x = c3;
                D = *p;
                D++;
                if (x > D)
                    x = D;
                *(p++) = x;
            }
            /* lower triangle sentinel */
            if (i <= half) {
                size_t c3 = --D + (char1 != *char2p);
                x++;
                if (x > c3)
                    x = c3;
                *p = x;
            }
        }

        i = *end;
        return i;
    }


    static int levenshtein(const void *data_point_1v, const void *data_point_2v, const void *datav) {
        const LevDataPoint* data_point_1 = reinterpret_cast<const LevDataPoint*>(data_point_1v);
        const LevDataPoint* data_point_2 = reinterpret_cast<const LevDataPoint*>(data_point_2v);
        const char* data = reinterpret_cast<const char*>(datav);
        const char* str1 = data_point_1->index >= 0 ? data + data_point_1->index : data_point_1->data;
        const char* str2 = data_point_2->index >= 0 ? data + data_point_2->index : data_point_2->data;

        return lev_edit_distance(strlen(str1), str1, strlen(str2), str2);
    }


    // Levenshtein distance space
    class LevSpace : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        char* data_;

    public:
        LevSpace(char* data) {
            fstdistfunc_ = levenshtein;
            data_ = data;
        }

        size_t get_data_size() {
            return sizeof(LevDataPoint);
        }

        DISTFUNC<int> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return data_;
        }

        ~LevSpace() {}
    };

}
