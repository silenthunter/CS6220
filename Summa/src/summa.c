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
	return a > b ? a : b;
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

	if(rank >= procGridX * procGridY) return;//Why would this happen?
			
	int proc_x = rank / procGridY;
	int proc_y = rank % procGridY;
	
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
	
	int rowRanks[procGridX];
	int colRanks[procGridY];
	
	//Set up communication
	MPI_Comm_group(MPI_COMM_WORLD, &origGrp);
	for(i = 0; i < procGridX; i++)
	{
		rowRanks[i] = proc_y + procGridY * i;
		if(rowRanks[i] >= procGridX * procGridY) printf("(%d)row error: %d\n", rank, rowRanks[i]);
	}
	for(i = 0; i < procGridY; i++)
	{
		colRanks[i] = proc_x * procGridY + i;
		if(colRanks[i] >= procGridX * procGridY) printf("(%d)col error: %d = %d * %d + %d\n", rank, colRanks[i], proc_x, procGridY, i);
	}
		
	MPI_Group_incl(origGrp, procGridX, rowRanks, &rowGrp);
	MPI_Group_incl(origGrp, procGridY, colRanks, &colGrp);
	
	if(MPI_Comm_create(MPI_COMM_WORLD, rowGrp, &rowComm) != MPI_SUCCESS) printf("Comm creation error\n");
	if(MPI_Comm_create(MPI_COMM_WORLD, colGrp, &colComm) != MPI_SUCCESS) printf("Comm creation error\n");
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	int rowRank, colRank;
	if(MPI_Group_rank(rowGrp, &rowRank) != MPI_SUCCESS) printf("ERROR %d\n", rank);
	if(MPI_Group_rank(colGrp, &colRank) != MPI_SUCCESS) printf("ERROR %d\n", rank);
	
	
	int memNum = pb > blockSizeK ? pb : blockSizeK;
	double *bufA = (double*)malloc(sizeof(double) * memNum * blockSizeM);
	double *bufB = (double*)malloc(sizeof(double) * memNum * blockSizeN);
	int pbPos = 0;
	int datRecv = 0;
	int datRecvMax = 0;
	double bufTemp[pb * blockSizeN];
	
	//Loop through blocks
	for(i = 0; i < max(procGridX, procGridY); i++)
	{
		while(datRecvMax < blockSizeM * blockSizeN)
		{
				
			//Zero the buffers
			memset(bufA, 0, sizeof(double) * blockSizeK * blockSizeM);
			memset(bufB, 0, sizeof(double) * blockSizeK * blockSizeN);
			
			//Row is self
			if(i == rowRank)
				memcpy(&bufA[pbPos * blockSizeM], &Ablock[datRecv], sizeof(double) * pb * blockSizeM);
			
			//Col is self
			if(i == colRank)
			{
				for(j = 0; j < pb * blockSizeN; j++)
				{
					bufTemp[j] = Bblock[pbPos + j / blockSizeK + (j % blockSizeK) * blockSizeK];
				}
			}
			
			if(i < procGridX) MPI_Bcast(&bufA[datRecv], pb * blockSizeM, MPI_DOUBLE, i, rowComm);
			if(i < procGridY)
			{
				MPI_Bcast(bufTemp, pb * blockSizeN, MPI_DOUBLE, i, colComm);
				for(j = 0; j < pb * blockSizeN; j++)
				{
					bufB[pbPos + j / blockSizeK + (j % blockSizeK) * blockSizeK] = bufTemp[j];
				}
			}
			
			/*if(rank == 0){
			printf("------------\n");
			printf("------------\n");
			print_matrixx(bufA, blockSizeK, blockSizeM);
			printf("------------\n");
			print_matrixx(bufB, blockSizeN, blockSizeK);
			printf("------------\n");
			printf("------------\n");}//*/
			
			datRecv += pb * blockSizeM;//This only applies to rows
			datRecvMax += pb * max(blockSizeM, blockSizeN);
			pbPos += pb < blockSizeK ? pb: blockSizeK;
			local_mm(blockSizeM, blockSizeN, blockSizeK, 1, bufA, blockSizeM, bufB, blockSizeK, 1, Cblock, blockSizeM);
		}
		datRecv = 0;//= datRecvB = 0;
		datRecvMax = 0;
		if(pbPos >= pb)
			pbPos = 0;
	}

	free(bufA);
	free(bufB);
	
	MPI_Barrier(MPI_COMM_WORLD);

}
