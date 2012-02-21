/* 
 * Georgia Tech HPC Class Project - MPI
 */

#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int gthpc_Allreduce (void *sbuf, void *rbuf, int count, MPI_Op op,
		      MPI_Comm comm);

int
main (int ac, char *av[])
{
  int i;
  int rank = -1;
  int count = 100;
  const int maxsize = 256 * 256;
  const int vsize = 1;
  double start, stop;
  double sbuf[maxsize];
  double rbuf[maxsize];

  MPI_Init (&ac, &av);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  for (i = 0; i < maxsize; i++)
    {
      sbuf[i] = (i + 1) * 1.0;
    }

  start = MPI_Wtime ();
  for (i = 0; i < count; i++)
    {
      gthpc_Allreduce (sbuf, rbuf, vsize, MPI_SUM, MPI_COMM_WORLD);
    }
  stop = MPI_Wtime ();

  if (rank == 0)
    {
      printf ("Runtime = %g\n"
	      "Calls = %d\n"
	      "Time per call = %g us\n",
	      stop - start, count, 1e6 * (stop - start) / count);
    }

  MPI_Finalize ();
  return 0;
}
