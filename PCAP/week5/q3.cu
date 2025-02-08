#include <stdio.h>
#include <cuda_runtime.h>

__global__ void computeSine(float *ang, float *val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        val[idx] = sinf(ang[idx]); 
    }
}

int main() {
    int n = 256;  

    float *h_ang = (float *)malloc(n * sizeof(float));
    float *h_val = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < 5; ++i) {
        h_ang[i] = i * 2.0f * 3.14159f / 5; 
    }

    float *d_ang, *d_val;
    cudaMalloc((void**)&d_ang, n * sizeof(float));
    cudaMalloc((void**)&d_val, n * sizeof(float));

    cudaMemcpy(d_ang, h_ang, n * sizeof(float), cudaMemcpyHostToDevice);

    computeSine<<<ceil(n/256.0), 256>>>(d_ang, d_val, n);

    cudaMemcpy(h_val, d_val, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sine values of the ang:\n");
    for (int i = 0; i < 5; ++i) {
        printf("sin(%.2f) = %.4f\n", h_ang[i], h_val[i]);
    }

    cudaFree(d_ang);
    cudaFree(d_val);
    free(h_ang);
    free(h_val);

    return 0;
}