
#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void oddEven(int *A, int n){
    int idx= blockIdx.x*blockDim.x+threadIdx.x;
    if(idx%2!=0 && idx+1<n){
        if(A[idx]>A[idx+1]) swap(&A[idx],&A[idx+1]);
    }
    if (idx % 2 == 0 && idx + 1 < n) {
        if (A[idx]>A[idx+1]) swap(&A[idx], &A[idx + 1]);
        }
}
int main(){
    int n;
    // Take the array size as input
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int *arr, *out;

    // Allocate memory on host
    arr = (int*)malloc(n * sizeof(int));
    out = (int*)malloc(n * sizeof(int));

    // Take array elements as input
    printf("Enter the elements of the array:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    // Declare device pointers
    int *d_arr, *d_out;

    // Allocate memory on device
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    // Copy input array from host to device
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to perform the odd-even transposition sort
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks to cover all elements

    for (int i = 0; i < n; i++) {  // Iterate for n passes (or a reasonable number)
        oddEven<<<numBlocks, blockSize>>>(d_arr, n);

        // Synchronize after each pass to ensure all threads are done
        cudaDeviceSynchronize();
    }

    // Copy the sorted array back to host
    cudaMemcpy(out, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted Array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");

    // Free memory on device and host
    cudaFree(d_arr);
    cudaFree(d_out);
    free(arr);
    free(out);
}