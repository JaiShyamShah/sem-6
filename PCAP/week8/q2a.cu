#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define N 3
__global__ void multiplyRow(int *A, int *B, int *C, int wA, int wB){
    int RidA=threadIdx.x;
    int sum;
    for(int CidB=0; CidB<wB; CidB++){
        sum=0;
        for(int k=0; k<wA; k++){
            sum+=A[RidA*wA+k]*B[k*wB+CidB];
        }
        C[RidA*wB+CidB]=sum;
    }
}

__global__ void multiplyCol(int *A, int *B, int *C, int hA, int wA){
    int CidB=threadIdx.x;
    int wB=blockDim.x;
    int sum;
    for(int RidA=0; RidA<hA; RidA++){
        sum=0;
        for(int k=0; k<wA; k++){
            sum+=A[RidA*wA+k]*B[k*wB+CidB];
        }
        C[RidA*wB+CidB]=sum;
    }
}

__global__ void multiplyEle(int *A, int *B, int *C, int wA){
    int RidA=threadIdx.y;
    int CidB=threadIdx.x;
    int wB=blockDim.x;
    int sum;
    for(int k=0; k<wA; k++){
        sum+=A[RidA*wA+k]*B[k*wB+CidB];
    }
    C[RidA*wB+CidB]=sum;
}

int main(){
    int h_A[N][N], h_B[N][N], h_C[N][N]; 


    printf("Enter elements of matrix A (3x3):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &h_A[i][j]);
        }
    }


    printf("Enter elements of matrix B (3x3):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("B[%d][%d]: ", i, j);
            scanf("%d", &h_B[i][j]);
        }
    }

    int *d_A, *d_B, *d_C; 
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));
    cudaMalloc((void**)&d_C, N * N * sizeof(int));


    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with one block and N threads in each dimension (for 3x3 matrices)
    multiplyRow<<<1, N>>>(d_A, d_B, d_C, N, N);

    // Copy the result matrix from device to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Matrix C (Result of A * B):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

