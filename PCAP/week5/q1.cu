#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void add_vectors(int *a, int *b, int *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; 
    if (i < n) {  // bounds checking to avoid accessing out of bounds
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    int n = 256;  // Length of the vectors
    int *h_A = (int *)malloc(n * sizeof(int));
    int *h_B = (int *)malloc(n * sizeof(int));
    int *h_C = (int *)malloc(n * sizeof(int));

    // Initialize vectors A and B
    for (int i = 0; i < n; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    int *d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void**)&d_A, n * sizeof(int));
    cudaMalloc((void**)&d_B, n * sizeof(int));
    cudaMalloc((void**)&d_C, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and n threads
    add_vectors<<<1, n>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Results-1: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Launch kernel with n blocks and 1 thread per block
    add_vectors<<<n, 1>>>(d_A, d_B, d_C, n);
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Results-2: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Launch kernel with dynamic grid and block size
    dim3 dimGrid(ceil(n / 256.0), 1, 1);  // Dynamically calculate the number of blocks
    dim3 dimBlock(256, 1, 1);  // Block size of 256 threads
    add_vectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    printf("Results-3: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_C[i]);
    }
    printf("\n");

    // Free memory on the device and host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
