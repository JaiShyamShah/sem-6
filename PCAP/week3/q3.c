#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

int is_non_vowel(char c) {
    // Check if the character is a non-vowel (case-insensitive)
    char lower_c = tolower(c);
    return !(lower_c == 'a' || lower_c == 'e' || lower_c == 'i' || lower_c == 'o' || lower_c == 'u' || lower_c == ' ');
}

int main(int argc, char* argv[]) {
    int rank, size, N, local_non_vowels = 0, total_non_vowels = 0;
    char str[1000];                // Static array for the string
    int str_len;
    int portion_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a string: ");
        fgets(str, sizeof(str), stdin);  // Ensure the buffer size is big enough

        // Remove newline character at the end of the string if present
        str_len = strlen(str);
        printf("String length is %d, dividing it evenly among %d processes.\n", str_len, size);
    }

    // Broadcast the length of the string to all processes
    MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the portion size for each process
    portion_size = str_len / size;

    // Each process needs a buffer to hold its portion of the string
    char local_str[portion_size];  // Buffer for each process, including null-terminator

    // Scatter the string portions to all processes
    MPI_Scatter(str, portion_size, MPI_CHAR, local_str, portion_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Count non-vowel characters in the local portion of the string
    for (int i = 0; i < portion_size; i++) {
        if (is_non_vowel(local_str[i])) {
            local_non_vowels++;
        }
    }

    // Gather the local counts from all processes to the root
    int recv_buffer[size];  // Array to gather non-vowel counts from all processes
    MPI_Gather(&local_non_vowels, 1, MPI_INT, recv_buffer, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process calculates and prints the result
    if (rank == 0) {
        // Print the non-vowels found by each process
        printf("Non-vowels found by each process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d found %d non-vowels.\n", i, recv_buffer[i]);
        }

        // Calculate the total number of non-vowels
        total_non_vowels = 0;
        for (int i = 0; i < size; i++) {
            total_non_vowels += recv_buffer[i];
        }
        printf("Total number of non-vowels in the string: %d\n", total_non_vowels);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
