/*
 * Georgia Tech HPC Class Project - MPI
 */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>


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
	int rank, threads, i;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &threads);
	MPI_Status status;
	//int sendSize = ceil((float)count / threads);

	//if(rank == threads - 1) sendSize -= count % threads;


	//Dimension d should be log_2(count)
	int d = log2(count);
	
	//Psuedo for All-to-all broadcast d-dimensional hypercube
	for(i = 0; i < d - 1; i++)
	{
		int partner = rank ^ (int)pow(2, i);
		//send/receive partner;
		MPI_Send(sbuf, 1, MPI_DOUBLE, partner, 0, comm);
		MPI_Recv(rbuf, 1, MPI_DOUBLE, partner, 0, comm, &status);

		if(op == MPI_SUM)
		{
		}
		else if(op == MPI_MAX)
		{
		}
		else
		{
		}
		//result += receive
	}

  return 0;
}
