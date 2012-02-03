#include <omp.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <stdlib.h>

using std::cout;
using std::endl;

void ComputeMatrix(double* A, double* B, double* C, int dim)
{
	int i, j, k;

	#if defined SINGLE || defined DOUBLE || defined TRIPLE
	#pragma omp parallel for
	#endif
	for(i = 0; i < dim * dim; i++)
	{
		C[i] = 0;
		#if defined DOUBLE || defined TRIPLE
		#pragma omp parallel for
		#endif
		for(j = 0; j < dim; j++)
		{
			#ifdef TRIPLE
			#pragma omp parallel for
			#endif
			for(k = 0; k < dim; k++)
			{
				C[i] += A[j] * B[k];
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


	clock_t start, end;

	start = clock();
	ComputeMatrix(A, B, C, dim);
	end = clock();

	cout << "Time: " << end - start << endl;
}
