/**
 *  \file summa.c
 *  \brief Implementation of Scalable Universal 
 *    Matrix Multiplication Algorithm for Proj1
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>

#include "local_mm.h"

#define max(a, b) (a > b ? a : b)

void printMatrix_(double *mat, int x, int y)
{
	int i, j;
	for(i = 0; i < y; i++)
	{
		for(j = 0; j < x; j++)
			printf("%1.0f ", mat[x * j + i]);
		printf("\n");
	}

	printf("\n");
	printf("\n");
	printf("\n");
}

/**
 * Distributed Matrix Multiply using the SUMMA algorithm
 *  Computes C = A*B + C
 * 
 *  This function uses procGridX times procGridY processes
 *   to compute the product
 *  
 *  A is a m by k matrix, each process starts
 *	with a block of A (aBlock) 
 *  
 *  B is a k by n matrix, each process starts
 *	with a block of B (bBlock) 
 *  
 *  C is a n by m matrix, each process starts
 *	with a block of C (cBlock)
 *
 *  The resulting matrix is stored in C.
 *  A and B should not be modified during computation.
 * 
 *  Ablock, Bblock, and CBlock are stored in
 *   column-major format  
 *
 *  pb is the Panel Block Size (Rows sent?)
 **/
void summa(int m, int n, int k, double *Ablock, double *Bblock, double *Cblock,
		int procGridX, int procGridY, int pb) {

	int rank, i, j, x;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank >= procGridX * procGridY) return;//Too many processes assigned
	/*if(rank == 0)
	for(i = 0; i < m * k; i++)
		printf("%1.0f\n", Bblock[i]);
	return;*/

	int procY = rank % procGridY;
	int procX = rank / procGridY;

	int rows[procGridY];
	int cols[procGridX];

	//Find out which processes to broadcast to
	for(i = 0; i < procGridY; i++)
		rows[i] = i * procGridY + procY;
	for(i = 0; i < procGridX; i++)
		cols[i] = procX * procGridY + i;

	//Create comm groups
	MPI_Group orig, rowGrp, colGrp;
	MPI_Comm_group(MPI_COMM_WORLD, &orig);
	MPI_Group_incl(orig, procGridY, rows, &rowGrp);
	MPI_Group_incl(orig, procGridX, cols, &colGrp);

	//Create communicators
	MPI_Comm rowComm, colComm;
	MPI_Comm_create(MPI_COMM_WORLD, rowGrp, &rowComm);
	MPI_Comm_create(MPI_COMM_WORLD, colGrp, &colComm);

	//Find dimensions for the block
	int dimM = m / procGridY;
	int dimN = n / procGridY;//TODO: Make sure this is correct
	int dimK = k / procGridX;//K seems to divide the same for each block

	//Allocate buffers for the received data
	int rowsReq = pb > dimK ? pb : dimK;
	double *bufA = (double*)malloc(sizeof(double) * rowsReq * dimM);
	double *bufB = (double*)malloc(sizeof(double) * rowsReq * dimN);
	double tmpB[pb * dimN];//Temporary buffer for storing row-major

	for(i = 0; i < max(procGridX, procGridY); i++)
	{
		for(j = 0; j < dimK; j += pb)
		{

			memset(bufA, 0, sizeof(double) * dimM * rowsReq);
			memset(bufB, 0, sizeof(double) * dimN * rowsReq);

			memcpy(&bufA[j * dimM], &Ablock[j * dimM], sizeof(double) * dimM * pb);
			for(x = 0; x < dimN * pb; x++)
				tmpB[x] = Bblock[(x % dimN) * dimN + j + x / dimN];

			MPI_Bcast(&bufA[j * dimM], pb * dimM, MPI_DOUBLE, i, rowComm);
			MPI_Bcast(tmpB, pb * dimN, MPI_DOUBLE, i, colComm);

			for(x = 0; x < dimN * pb; x++)
				bufB[(x % dimN) * dimN + j + x / dimN] = tmpB[x];
			
			local_mm(dimM, dimN, dimK, 1, bufA, dimM, bufB, dimK, 1, Cblock, dimM);

			/*if(rank == 0)printf("---------------\n");
			//if(rank == 0)printMatrix_(bufA, pb, dimM);
			//if(rank == 0)printMatrix_(bufB, dimN, pb);
			if(rank == 0)printMatrix_(Cblock, dimN, dimM);
			if(rank == 0)printf("---------------\n");*/
		}
	}
	free(bufA);
	free(bufB);

	
	MPI_Barrier(MPI_COMM_WORLD);

}
