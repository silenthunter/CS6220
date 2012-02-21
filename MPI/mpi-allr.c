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

  return 0;
}
