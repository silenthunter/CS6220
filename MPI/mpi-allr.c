/*
 * Georgia Tech HPC Class Project - MPI
 */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define max(a, b) (a > b  ? a : b)
#define min(a, b) (a < b  ? a : b)

/* 
 * MPI Specification: int MPI_Allreduce(void *sbuf, void *rbuf, int
 * count, MPI_Datatype dtype, MPI_Op op, MPI_Comm comm)
 */


/*
 * int gthpc_Allreduce(void *sbuf, void *rbuf, int count, MPI_Op op,
 * MPI_Comm comm)
 **/
int
gthpc_Allreduce (void *sbuf, void *rbuf, int count, MPI_Op op, MPI_Comm comm)
{

	/* write your own code using MPI point to point ops to implement the gthpc_Allreduce here... 
	 *
	 * the equivalent call using the MPI collective is 
	 *
	 * MPI_Allreduce (sbuf, rbuf, count, MPI_DOUBLE, op, comm);
	 */
	int rank, threads, i, j;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &threads);
	MPI_Status status;
	double tempBuff[count];
	//int sendSize = ceil((float)count / threads);

	//if(rank == threads - 1) sendSize -= count % threads;


	//Dimension d should be log_2(count)
	int d = log2(threads);
	for(j = 0; j < count; j++) ((double*)rbuf)[j] = ((double*)sbuf)[j];

	int sendCounter = 0;
	
	//Psuedo for All-to-all broadcast d-dimensional hypercube
	for(i = 0; i < d; i++)
	{
		int partner = rank ^ (int)pow(2, i);

		//printf("%d --> %d\n", rank, partner);

		//send/receive partner;
		if(rank & (int)pow(2, i)){
			MPI_Send(rbuf, count, MPI_DOUBLE, partner, 0, comm);
			MPI_Recv(&tempBuff, count, MPI_DOUBLE, partner, 0, comm, &status);
		} else {
			MPI_Recv(&tempBuff, count, MPI_DOUBLE, partner, 0, comm, &status);
			MPI_Send(rbuf, count, MPI_DOUBLE, partner, 0, comm);
		}
		sendCounter++;

		for(j = 0; j < count; j++)
		{
		if(op == MPI_SUM)
			((double*)rbuf)[j] += tempBuff[j];
		else if(op == MPI_MAX)
			((double*)rbuf)[j] = max(((double*)rbuf)[j], tempBuff[j]);
		else
			((double*)rbuf)[j] = min(((double*)rbuf)[j], tempBuff[j]);
		}
	}
	#ifdef VERBOSE
	if(rank == 0) printf("%d sends per node with %d nodes\n", sendCounter, threads);
	#endif

  return 0;
}
