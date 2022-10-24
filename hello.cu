%%writefile hw1.cu


#include <stdio.h>

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
            printf("%s device name:\n", deviceProp.name );
            printf("the maximum number of thread blocks:\n");    
            printf("the maximum number of threads per block at the beginning:\n");
            }
    }
}

int main(void)
{
    

    getInformationAboutSystem();
    
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
}   