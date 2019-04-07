#define ARR_DIM 512 * 512 * 512
#define MAT_DIM 512 
#define MAX_NUM 128

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

const float sec_const = 1000000.0;

int deinit(int32_t *&array){

    delete[] array;

    return 0;
}

int init_zero(int32_t *&array, int32_t length){
    array = new int32_t[length];

    for(int i = 0; i < length; i++)
        array[i] = 0;

    return 0;
}

int init_rand(int32_t *&array, int32_t length, int32_t max_num){

    init_zero(array, length);

    srand((unsigned)time(nullptr));

    for(size_t i = 0; i < length; i++)
    {
        array[i] = rand() % max_num;
    }
    
    return 0;
}

int raw_vec_multadd(int32_t *&v1, int32_t *&v2, int32_t *&v3, int32_t *&v4, int32_t length){

    int32_t *r;
    init_zero(r, length);

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    
    start_t = clock();

    for(size_t i = 0; i < length; i++)
    {
        r[i] = v1[i] * v2[i] + v3[i] * v4[i];
    }
    
    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("raw\t %.2fs\n", clock_delta_sec);

    deinit(r);
    return 0;
}

int vec_vec_multadd(int32_t *&v1, int32_t *&v2, int32_t *&v3, int32_t *&v4, int32_t length){

    int32_t *r, *m1, *m2;
    init_zero(r, length);
    init_zero(m1, length);
    init_zero(m2, length);

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    
    __m256i rV1, rV2, rV3, rV4, rM1, rM2, rR;

    start_t = clock();

    for(size_t i = 0; i < length; i += 8)
    {
        rV1 = _mm256_load_si256((__m256i *)&v1[i]);
        rV2 = _mm256_load_si256((__m256i *)&v2[i]);
        rV3 = _mm256_load_si256((__m256i *)&v3[i]);
        rV4 = _mm256_load_si256((__m256i *)&v4[i]);
        rM1 = _mm256_mullo_epi32(rV1, rV2); // might be overflow
        rM2 = _mm256_mullo_epi32(rV3, rV4);
        rR = _mm256_add_epi32(rM1, rM2); 
        _mm256_store_si256((__m256i *)&r[i],rR);
    }

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("vec\t %.2fs\n", clock_delta_sec);

    deinit(r);
    return 0;
}

int deinit(int32_t **&array, int32_t side){

    if(array == nullptr)
        return 1;

    for(int i = 0; i < side; i++)
        delete[] array[i];

    delete[] array;

    return 0;
}

int init_zero(int32_t **&array, int32_t side){

    array = new int32_t*[side];
    
    for(int i = 0; i < side; i++){
        array[i] = new int32_t[side];

        for(int j = 0; j < side; j++)
            array[i][j] = 0;
    }
    
    return 0;
}

int init_rand(int32_t **&array, int32_t side, int32_t max_num){

    init_zero(array, side);

    srand((unsigned)time(nullptr));

    for(int i = 0; i < side; i++)
        for(int j = 0; j < side; j++)
            array[i][j] = rand() % max_num;

    return 0;
}

int raw_mat_mult(int32_t **&m1, int32_t **&m2, int32_t side){

    int32_t** r;
    init_zero(r, side);

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    
    start_t = clock();

    for(int i = 0; i < side; i++){
        for(int j = 0; j < side; j++){
            for(int k = 0; k < side; k++){
                r[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("raw\t %.2fs\n", clock_delta_sec);

    deinit(r, side);
    return 0;
}

int blas_mat_mult(int32_t **&m1, int32_t **&m2, int32_t side){
    int32_t** r;
    init_zero(r, side);

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    
    start_t = clock();

    // for(int i = 0; i < side; i++){
    //     for(int j = 0; j < side; j++){
    //         for(int k = 0; k < side; k++){
    //             r[i][j] += m1[i][k] * m2[k][j];
    //         }
    //     }
    // }

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("blas\t %.2fs\n", clock_delta_sec);

    deinit(r, side);
    return 0;
}

int vec_mat_mult(int32_t **&m1, int32_t **&m2, int32_t side){

    int32_t** r;
    init_zero(r, side);

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    
    start_t = clock();

    // for(int i = 0; i < side; i++){
    //     for(int j = 0; j < side; j++){
    //         for(int k = 0; k < side; k++){
    //             r[i][j] += m1[i][k] * m2[k][j];
    //         }
    //     }
    // }

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("vec\t %.2fs\n", clock_delta_sec);

    deinit(r, side);
    return 0;
}

int main(){

    printf("Vectors mult and add\n");

    int32_t *v1, *v2, *v3, *v4; // since memory is allocated dynamically, all arrays are aligned

    init_rand(v1, ARR_DIM, MAX_NUM);
    init_rand(v2, ARR_DIM, MAX_NUM);
    init_rand(v3, ARR_DIM, MAX_NUM);
    init_rand(v4, ARR_DIM, MAX_NUM);

    raw_vec_multadd(v1, v2, v3, v4, ARR_DIM);
    raw_vec_multadd(v1, v2, v3, v4, ARR_DIM);
    raw_vec_multadd(v1, v2, v3, v4, ARR_DIM);
    vec_vec_multadd(v1, v2, v3, v4, ARR_DIM);
    vec_vec_multadd(v1, v2, v3, v4, ARR_DIM);
    vec_vec_multadd(v1, v2, v3, v4, ARR_DIM);

    deinit(v1);
    deinit(v2);
    deinit(v3);
    deinit(v4);

    // printf("Matrices mult\n");

    // int32_t **m1, **m2;

    // init_rand(m1, MAT_DIM, MAX_NUM);
    // init_rand(m2, MAT_DIM, MAX_NUM);

    // raw_mat_mult(m1, m2, MAT_DIM);
    // blas_mat_mult(m1, m2, MAT_DIM);
    // vec_mat_mult(m1, m2, MAT_DIM);

    // deinit(m1, MAT_DIM);
    // deinit(m2, MAT_DIM);

    return 0;
}