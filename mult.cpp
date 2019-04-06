#define ARR_DIM 512
#define ARR_MAX_NUM 128

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const float sec_const = 1000000.0;

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

int deinit(int32_t **&array, int32_t side){

    if(array == nullptr)
        return 0;

    for(int i = 0; i < side; i++)
        delete[] array[i];

    delete[] array;

    return 0;
}

int raw_mult(int32_t **&m1, int32_t **&m2, int32_t side){

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

int vec_mult(int32_t **&m1, int32_t **&m2, int32_t side){

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

    int32_t **m1, **m2;

    init_rand(m1, ARR_DIM, ARR_MAX_NUM);
    init_rand(m2, ARR_DIM, ARR_MAX_NUM);

    raw_mult(m1, m2, ARR_DIM);
    vec_mult(m1, m2, ARR_DIM);

    deinit(m1, ARR_DIM);
    deinit(m2, ARR_DIM);

    return 0;
}