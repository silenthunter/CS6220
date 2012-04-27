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

int max(int a, int b)
{
	return a < b ? a : b;
}

void print_matrixx(double* a, int b, int c)
{
	int i, j;
	for(i = 0; i < c; i++)
	{
		for(j = 0; j < b; j++)
		{
			printf("%1.0f ", a[j * c + i]);
		}
		printf("\n");
	}
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
			
	int rank, i, j;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get processor id
	MPI_Comm rowComm, colComm;
	MPI_Group origGrp, rowGrp, colGrp;

	if(rank >= procGridX * procGridY) return;//Too many processes assigned
			
	int proc_x = rank % procGridX;
	int proc_y = rank / procGridX;
	
	int blockSizeK = k / procGridY;
	int blockSizeM = m / procGridX;
	int blockSizeN = n / procGridY;
	
	if(rank == 0 && 0)
	{
		for(i = 0; i < 4; i++)
			printf("%f ", Ablock[i]);
		printf("\n");
		
		for(i = 0; i < 4; i++)
			printf("%f ", Bblock[i]);
		printf("\n");
	}
	
	int rowRanks[procGridY];
	int colRanks[procGridX];
	
	//Set up communication
	MPI_Comm_group(MPI_COMM_WORLD, &origGrp);
	for(i = 0; i < procGridY; i++)
		rowRanks[i] = proc_x + procGridX * i;
	for(i = 0; i < procGridX; i++)
		colRanks[i] = proc_y * procGridX + i;
		
	MPI_Group_incl(origGrp, procGridY, rowRanks, &rowGrp);
	MPI_Group_incl(origGrp, procGridX, colRanks, &colGrp);
	
	MPI_Comm_create(MPI_COMM_WORLD, rowGrp, &rowComm);
	MPI_Comm_create(MPI_COMM_WORLD, colGrp, &colComm);
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	int rowRank, colRank;
	MPI_Group_rank(rowGrp, &rowRank);
	MPI_Group_rank(colGrp, &colRank);
	
	int memNum = pb > blockSizeK ? pb : blockSizeK;
	double *bufA = (double*)malloc(sizeof(double) * memNum * blockSizeM);
	double *bufB = (double*)malloc(sizeof(double) * memNum * blockSizeN);
	int pbPos = 0;
	int datRecv = 0;
	
	//Loop through blocks
	for(i = 0; i < max(procGridX, procGridY); i++)
	{
		while(datRecv < blockSizeM * blockSizeK)
		{
				
			//Zero the buffers
			memset(bufA, 0, sizeof(double) * blockSizeK * blockSizeM);
			memset(bufB, 0, sizeof(double) * blockSizeK * blockSizeN);
			
			//Row is self
			if(i == rowRank)
				memcpy(&bufA[pbPos * blockSizeM], &Ablock[datRecv], sizeof(double) * pb * blockSizeM);
			
			double bufTemp[pb * blockSizeN];
			//Col is self
			if(i == colRank)
			{
				for(j = 0; j < pb * blockSizeN; j++)
				{
					bufTemp[j] = Bblock[pbPos + j / blockSizeK + (j % blockSizeK) * blockSizeK];
				}
			}
			
			MPI_Bcast(&bufA[datRecv], pb * blockSizeM, MPI_DOUBLE, i, rowComm);
			MPI_Bcast(bufTemp, pb * blockSizeN, MPI_DOUBLE, i, colComm);
			for(j = 0; j < pb * blockSizeN; j++)
			{
				bufB[pbPos + j / blockSizeK + (j % blockSizeK) * blockSizeK] = bufTemp[j];
			}
			
			/*printf("------------\n");
			printf("------------\n");
			print_matrixx(bufA, blockSizeK, blockSizeM);
			printf("------------\n");
			print_matrixx(bufB, blockSizeN, blockSizeK);
			printf("------------\n");
			printf("------------\n");//*/
			
			datRecv += pb * blockSizeM;
			pbPos += pb < blockSizeK ? pb: blockSizeK;//+= pb * blockSizeN;
			local_mm(blockSizeM, blockSizeN, blockSizeK, 1, bufA, blockSizeM, bufB, blockSizeK, 1, Cblock, blockSizeM);
		}
		datRecv = 0;
		if(pbPos >= pb)
			pbPos = 0;
		
		//local_mm(blockSizeM, blockSizeN, blockSizeK, 1, bufA, blockSizeM, bufB, blockSizeK, 1, Cblock, blockSizeM);
	}
	free(bufA);
	free(bufB);
	
	MPI_Barrier(MPI_COMM_WORLD);

}
