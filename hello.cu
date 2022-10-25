// %%writefile hw1.cu for Google Colab


#include <stdio.h>
#include <time.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
    
}

__host__ 
void getInformationAboutSystem(){
    int deviceNumber;
    cudaGetDeviceCount(&deviceNumber);
   
    for (int i = 0; i< deviceNumber; i++){
   
        cudaDeviceProp deviceProp;
        cudaError_t err =  cudaGetDeviceProperties (&deviceProp, i);
        
        if (!err) {
            printf("The device name: %s \n", deviceProp.name );
            printf("the maximum number of thread blocks: %d \n", deviceProp.maxBlocksPerMultiProcessor);
            printf("the maximum number of threads per block at the beginning: %d \n", deviceProp.maxThreadsPerBlock);
            }
    }
}

__host__ 
float* generateRandomElements(int N, float constant){

    float *arr;
    arr = (float*)malloc(N*sizeof(float));  
    if ( arr == NULL ){
        printf("Run out of memmory!\n");
        exit(1);
    }

    
    for (int i = 0; i<N; i++){
          arr[i] = ((float)rand()/RAND_MAX)* constant; //generate random float element for array
        //   printf("index %d: %f | ", i , arr[i]);
          
    }
    return arr;
}

int main(void)
{
    clock_t start, end;
    double time_used;

    int N;
    float A;
    printf("Enter a size for array: ");
    scanf("%d", &N);


    printf("Enter a scalar value: ");
    scanf("%f", &A);
    // printf("%f, %d", A, N);
    
    getInformationAboutSystem();
    

    float *x, *y, *d_x, *d_y;

    printf("X Vector is createad : \n");
    x = generateRandomElements(N, 1.0f);
    printf("\n--------------------------------------\n");
    printf("Y Vector is createad : \n");
    y = generateRandomElements(N, 2.0f);
    printf("\n--------------------------------------\n");

    start = clock();
    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<ceil(N/1024),1024>>>(N, A, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // printf("After SAXPY Y Vector is like following\n");
    // for (int i = 0; i < N; i++)
    //     printf("index %d: %f | ", i, y[i]);

    printf("%lf is total compile time on GPU. \n", time_used);
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}   
