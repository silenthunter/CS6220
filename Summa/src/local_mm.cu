/**
 *  \file local_mm.c
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "local_mm.h"
#include <cuda.h>
#include <thrust/transform_reduce.h>

extern __shared__ double sharedLines[];

/**
 *
 *  Local Matrix Multiply
 *   Computes C = alpha * A * B + beta * C
 *
 *
 *  Similar to the DGEMM routine in BLAS
 *
 *
 *  alpha and beta are double-precision scalars
 *
 *  A, B, and C are matrices of double-precision elements
 *  stored in column-major format 
 *
 *  The output is stored in C
 *  A and B are not modified during computation
 *
 *
 *  m - number of rows of matrix A and rows of C
 *  n - number of columns of matrix B and columns of C
 *  k - number of columns of matrix A and rows of B
 * 
 *  lda, ldb, and ldc specifies the size of the first dimension of the matrices
 *
 **/
__global__ void local_mm_device (const int m, const int n, const int k, const double alpha,
	  const double *A, const int lda, const double *B, const int ldb,
	  const double beta, double *C, const int ldc)
{
	double *sharedA = &sharedLines[0];
	double *sharedB = &sharedLines[k * n];
	int idx = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
	sharedA[idx] = A[idx];
	sharedB[idx] = B[idx];
	
	__syncthreads();//shared memory is done transfering
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int k_iter;
	double dotprod = 0.0;
	for (k_iter = 0; k_iter < k; k_iter++)
	{
		int a_index = (k_iter * lda) + row;	/* Compute index of A element */
		int b_index = (col * ldb) + k_iter;	/* Compute index of B element */
		dotprod += sharedA[a_index] * sharedB[b_index];	/* Compute product of A and B */
	}
	int c_index = (col * ldc) + row;
	C[c_index] = (alpha * dotprod) + (beta * C[c_index]);

	//__syncthreads();//Wait until each thread has their results in
}

static int deviceCount = 0;
static long globalMemory;
static long sharedMemory;

void initDeviceProperties()
{
	int count;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);//Assumes both devices are the same
	cudaGetDeviceCount(&count);

	deviceCount = count;
	globalMemory = prop.totalGlobalMem;
	sharedMemory = prop.sharedMemPerBlock;
}

extern "C" void local_mm (const int m, const int n, const int k, const double alpha,
	  const double *A, const int lda, const double *B, const int ldb,
	  const double beta, double *C, const int ldc)
{

if(deviceCount == 0) initDeviceProperties();

double *d_C[deviceCount];

for(int i = 0; i < deviceCount; i++)
{
	cudaSetDevice(i);
	int offset = k / deviceCount * i;
	double *d_A, *d_B;
	cudaMalloc((void**)&d_A, sizeof(double) * m * k / deviceCount);
	cudaMalloc((void**)&d_B, sizeof(double) * n * k / deviceCount);
	cudaMalloc((void**)&d_C[i], sizeof(double) * m * n / deviceCount);

	//Copy local array to device
	cudaMemcpy(d_A, &A[offset], sizeof(double) * m * k / deviceCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, &B[offset], sizeof(double) * m * k / deviceCount, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C[i], &C[offset], sizeof(double) * m * k / deviceCount, cudaMemcpyHostToDevice);

	//Find the number of rows that can fit in shared memory
	int rowMax = sharedMemory / (sizeof(double) * max(n,m));
	int rowMemRequired = rowMax * max(n,m) * sizeof(double);

	int blocks = k / rowMax;

	dim3 blocksInGrid(1, blocks, 1);
	dim3 threadsInBlock(rowMax, max(n,m));
	local_mm_device<<<blocksInGrid,threadsInBlock,rowMemRequired>>>(m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C[i], ldc);

}
//Gather data from async launches
for(int i = 0; i < deviceCount; i++)
{
	int offset = k / deviceCount * i;
	cudaSetDevice(i);
	cudaMemcpy(&C[offset], d_C[i], sizeof(double) * m * k / deviceCount, cudaMemcpyDeviceToHost);
}


}

