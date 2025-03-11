#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// CUDA kernel to copy a string in progressively smaller sizes
__global__ void copyStringProgressively(char* d_result, const char* d_input, int str_len, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread copies progressively smaller versions of the string
    if (idx < n) {
        // Calculate the number of characters to copy for this index
        int length_to_copy = str_len - idx;
        
        // Calculate the starting index in the result array for this thread
        int offset = idx * str_len;

        // Copy only the first `length_to_copy` characters into the result array
        for (int i = 0; i < length_to_copy; ++i) {
            d_result[offset + i] = d_input[i];
        }

        // Ensure the next part is null-terminated
        if (length_to_copy < str_len) {
            d_result[offset + length_to_copy] = '\0';
        }
    }
}

int main() {
    const char* input_string = "ABCDEF";  // Example input string
    int n = strlen(input_string);  // Number of progressively smaller versions of the string

    int str_len = strlen(input_string);  // Length of the string
    int result_len = str_len * n;  // Total length of the result string

    // Allocate memory on the device
    char* d_input;
    char* d_result;
    cudaMalloc((void**)&d_input, str_len * sizeof(char));
    cudaMalloc((void**)&d_result, result_len * sizeof(char));

    // Copy input string to device memory
    cudaMemcpy(d_input, input_string, str_len * sizeof(char), cudaMemcpyHostToDevice);

    // Define the number of threads and blocks
    int threadsPerBlock = 256;  // Adjust as needed
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // Ensure enough blocks to cover `n` copies

    // Launch kernel to copy progressively smaller strings
    copyStringProgressively<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_input, str_len, n);

    // Allocate memory on the host for the result string
    char* h_result = (char*)malloc(result_len + 1);  // +1 for null terminator

    // Copy the result back to host memory
    cudaMemcpy(h_result, d_result, result_len * sizeof(char), cudaMemcpyDeviceToHost);

    // Null-terminate the result string
    h_result[result_len] = '\0';

    // Output the result
    printf("Result after progressively shortening the string:\n");
    for (int i = 0; i < n; ++i) {
        int length_to_print = str_len - i;
        for (int j = 0; j < length_to_print; ++j) {
            printf("%c", h_result[i * str_len + j]);
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_result);

    // Free host memory
    free(h_result);

    return 0;
}