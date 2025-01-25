#include <stdio.h>
#include <mpi.h>
int main(int argc, char** argv) {
    int rank, size,n,i;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        printf("Enter a number to send: ");
        scanf("%d",&n);
        for(int i=1;i<size;i++){
        MPI_Send(&n, 1, MPI_INT, i, i, MPI_COMM_WORLD);}
    } 
    else{
        MPI_Recv(&n, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);
        printf("Number received by rank %d: %d\n", rank,n);
    }
    MPI_Finalize();
    return 0;
}
