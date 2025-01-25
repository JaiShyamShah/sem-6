#include <mpi.h>
#include <stdio.h>

int main(int argc, char*argv[]){
	int rank,size,n;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Status status;

	if(rank == 0){
		printf("Enter value \n");
		scanf("%d",&n);

		MPI_Send(&n,1,MPI_INT,1,1,MPI_COMM_WORLD);
		MPI_Recv(&n,1,MPI_INT,size-1,1,MPI_COMM_WORLD,&status);
		printf("Rank %d Received %d \n",rank,n);
	}
	else{
		MPI_Recv(&n,1,MPI_INT,rank-1,1,MPI_COMM_WORLD,&status);
		printf("Rank %d Received %d \n",rank,n);
		n++;
		MPI_Send(&n,1,MPI_INT,(rank+1) % size,1,MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
