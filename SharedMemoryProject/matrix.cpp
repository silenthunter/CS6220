#include <omp.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

using std::cout;
using std::endl;

void ComputeMatrix(double* A, double* B, double* C, int dim)
{
	int i, j, k;

	//#pragma omp parallel shared(A, B, C, dim)
	#if defined SINGLE || defined DOUBLE || defined TRIPLE
	#pragma omp parallel for schedule(static)
	#endif
	for(i = 0; i < dim; i++)
	{
		#if defined DOUBLE || defined TRIPLE
		#pragma omp parallel for schedule(static)
		#endif
		for(j = 0; j < dim; j++)
		{
			C[i * dim + j] = 0;
			#ifdef TRIPLE
			#pragma omp parallel for schedule(static)
			#endif
			for(k = 0; k < dim; k++)
			{
				C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}
		}
	}
}

int main(int argc, char* argv[])
{
	if(argc < 2) exit(-1);
	int dim = atoi(argv[1]);
	double* A = new double[dim * dim];
	double* B = new double[dim * dim];
	double* C = new double[dim * dim];


	timeval start, end;

	gettimeofday(&start, NULL);
	ComputeMatrix(A, B, C, dim);
	gettimeofday(&end, NULL);
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec;
	long endTime = end.tv_sec * 1000000 + end.tv_usec;
	double startTimeSec = (double)startTime / 1000000;
	double endTimeSec = (double)endTime / 1000000;

	cout << "Time: " << endTimeSec - startTimeSec << endl;
}
