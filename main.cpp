#define ARR_DIM 512 * 512 * 128
#define MAT_DIM 1024 // matrices 1024x1024
#define MAX_NUM 128

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

const float sec_const = 1000000.0;

int deinit(int32_t *&array){

    delete[] array;

    return 0;
}

int deinit(double *&array){

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

    start_t = clock();

    __m256i rV1, rV2, rV3, rV4, rM1, rM2, rR;

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

int mat_transpose(int32_t **&array, int32_t side){

    int32_t** new_array;

    init_zero(new_array, side);

    for(size_t i = 0; i < side; i++)
    {
        for(size_t j = 0; j < side; j++)
        {
            new_array[i][j] = array[j][i];
        }
    }
    
    deinit(array, side);

    array = new_array;

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
    
    double *r = new double[side * side];

    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    
    start_t = clock();

    double *A = new double[side * side];
    double *B = new double[side * side];

    for(size_t i = 0; i < side; i++)
    {
        for(size_t j = 0; j < side; j++)
        {
            A[i * side + j] = m1[i][j];
            B[i * side + j] = m2[i][j];
        }
        
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, side, side, side, 1, A, side, B, side, 0, r, side);

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("blas\t %.2fs\n", clock_delta_sec);

    deinit(r);
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

    mat_transpose(m2, side);

    __m256i rM1, rM2, rR;
    int32_t *s = new int32_t[8], sum = 0;

    for(size_t i = 0; i < side; i++){
        for(size_t j = 0; j < side; j++){
            for(size_t k = 0; k < side; k += 8){
                rM1 = _mm256_load_si256((__m256i *)&m1[i][k]);
                rM2 = _mm256_load_si256((__m256i *)&m2[j][k]);
                rR = _mm256_mullo_epi32(rM1, rM2); // might be overflow
                _mm256_store_si256((__m256i *)s,rR);

                sum = 0;
                for(size_t p = 0; p < 8; p++)
                    sum += s[p];
                r[i][j] += sum;
            }
        }
    }

    mat_transpose(m2, side);

    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("vec\t %.2fs\n", clock_delta_sec);

    deinit(r, side);
    return 0;
}

// find position of seq in str, return position, else -1
int raw_substr(const char str[], int32_t str_len, const char seq[], int32_t seq_len){

    for(size_t i = 0; i < str_len; i++)
    {

        for(size_t j = 0; j < seq_len; j++)
        {
            if(seq[j] != str[i+j])
                break;

            if(j + 1 == seq_len)
                return i;
        }
    }
    
    return -1;
}

// find position of seq in str, return position, else -1
int vec_substr(const char str[], int32_t str_len, const char seq[], int32_t seq_len){

    __m128i rT1, rT2, rT3;
    __m256i rA1, rA2, rR;
    __int128_t* result = new __int128_t[2];
    bool b1, b2;

    // so, the main idea is that we search in two subsequent iterations simultaneously by 16 chars/time
    //
    // it can be rewritten to search in 4 iterations by 8 chars (or 8/4), that is faster, but harder to read
    // I just wanted to show the approach

    for(int32_t i = 0; i < str_len; i+=2)
    {

        b1 = true; // assume that we found seq
        b2 = true;

        for(int32_t j = 0; j < seq_len; j+=16)
        {
            if(b1 || b2 == false)
                break;

            int32_t shift = (seq_len - j) > 16 ? 0 : 16 - (seq_len - j); // if we search by less then 16 chars, we will need to drop some bits later

            rT1 = _mm_loadu_si128((__m128i *)&str[i+j]);
            rT2 = _mm_loadu_si128((__m128i *)&str[i+j+1]);
            rT3 = _mm_load_si128((__m128i *)&seq[j]);

            rA1 = _mm256_loadu2_m128i((__m128i *)&rT2, (__m128i *)&rT1);
            rA2 = _mm256_loadu2_m128i((__m128i *)&rT3, (__m128i *)&rT3);

            rR = _mm256_xor_si256(rA1, rA2); // if chars are same, a XOR a == 0

            _mm256_storeu_si256((__m256i *)result, rR);

            if(result[0] << (shift * 8) != 0) // drop bits that are not significant
                b1 = false;

            if(result[1] << (shift * 8) != 0)
                b2 = false;

            if(b1 && j + 16 >= seq_len)
                return i;

            if(b2 && j + 16 >= seq_len)
                return i + 1;
        }
    }
    
    return -1;
}

int test_substr(const char str[], int32_t str_len, const char seq[], int32_t seq_len){
    clock_t start_t;
    clock_t end_t;
    clock_t clock_delta;
    double clock_delta_sec;
    
    start_t = clock();

    for(size_t i = 0; i < 10000000; i++)
    {
        int _ = raw_substr(str, str_len, seq, seq_len);
    }
    
    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("raw\t %.2fs\n", clock_delta_sec);

    start_t = clock();

    for(size_t i = 0; i < 10000000; i++)
    {
        int _ = vec_substr(str, str_len, seq, seq_len);
    }
    
    end_t = clock();
    clock_delta = end_t - start_t;
    clock_delta_sec = (double) (clock_delta / sec_const);
    printf("vec\t %.2fs\n", clock_delta_sec);

    return 0;
}

int main(){

    printf("\nVectors mult and add\n");

    int32_t *v1 __attribute__((aligned(8))), *v2 __attribute__((aligned(8))), *v3 __attribute__((aligned(8))), *v4 __attribute__((aligned(8)));

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

    printf("\nMatrices mult\n");

    int32_t **m1, **m2;

    init_rand(m1, MAT_DIM, MAX_NUM);
    init_rand(m2, MAT_DIM, MAX_NUM);

    raw_mat_mult(m1, m2, MAT_DIM);
    raw_mat_mult(m1, m2, MAT_DIM);
    raw_mat_mult(m1, m2, MAT_DIM);
    blas_mat_mult(m1, m2, MAT_DIM);
    blas_mat_mult(m1, m2, MAT_DIM);
    blas_mat_mult(m1, m2, MAT_DIM);
    vec_mat_mult(m1, m2, MAT_DIM);
    vec_mat_mult(m1, m2, MAT_DIM);
    vec_mat_mult(m1, m2, MAT_DIM);

    deinit(m1, MAT_DIM);
    deinit(m2, MAT_DIM);

    printf("\nShort strings\n");

    const char str[256] = "XYZAVKLRPZA"; 
    const char seq[16] __attribute__((aligned(16))) = "ZA";
    test_substr(str, strlen(str), seq, strlen(seq));

    const char str2[256] = "XYZAVKLRSGDFSDPZASFSGFXYZAVKLRPZAXYZSDFSAVKLRSFGDGDFCPZAXYZAVKLRSGDFSDPZASFSGFXYZAVKLRPZAXYZSDFSAVKLRSFGDGDFCPZASDFHJFKSDFKJSDFJKSDFJK"; 
    const char seq2[16] __attribute__((aligned(16))) = "FKSDFKJSDFJK";
    printf("Other strings:\n");
    test_substr(str2, strlen(str2), seq2, strlen(seq2));

    const char str3[256] = "DPZASFSGFXYZAVKLRPZAVKLRSFGDGDFCPZAXYZAVKLRSGDFSDPZASFSGFXYZAVKLRPZAXYZSDFSAVKLRSFGDGDFCPZASDKSDFKJSDFJKSDFJKXYZAVKLRSGDFSDPZASFSGFXYZAVKLRPZAXYZSDFSAVKLRSFGDGDFCPZAXYZAVKLRSGDFSDPZASFSGFXYZAVKLRPZAXYZSDFSAVKLRSFGDGDFCPZASDFHJFKSDFKJSDFJKSDFJKDFSFSFSFSFSD"; 
    const char seq3[32] __attribute__((aligned(16))) = "FKSDFKJSDFJKSDFJKDFSFSFSFSFSD";
    printf("Long strings:\n");
    test_substr(str3, strlen(str3), seq3, strlen(seq3));

    // int32_t pos_raw = raw_substr(str3, strlen(str3), seq3, strlen(seq3));
    // int32_t pos_vec = vec_substr(str3, strlen(str3), seq3, strlen(seq3));
    // printf("raw: First occurence at position #%i\n", pos_raw);
    // printf("vec: First occurence at position #%i\n", pos_vec);

    return 0;
}