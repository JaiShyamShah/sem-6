#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void SS(int *A, int* o, int n){
    int idx= blockIdx.x*blockDim.x+threadIdx.x;
    int data= A[idx], pos=0;
    if(idx>=n) return;
    for(int i=0; i<n;i++){
        if(A[i]< data || (A[i]==data && i<idx)){
            pos++;
        }
    }
    o[pos]=data;
}
int main(){
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int *arr, *out;

    // Allocate memory on host
    arr = (int*)malloc(n * sizeof(int));
    out = (int*)malloc(n * sizeof(int));
    
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

    // Launch kernel to perform the selection sort
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks to cover all elements
    SS<<<numBlocks, blockSize>>>(d_arr, d_out, n);

    // Synchronize the device to make sure all threads have finished
    cudaDeviceSynchronize();

    // Copy the sorted array back to host
    cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

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

    return 0;
}