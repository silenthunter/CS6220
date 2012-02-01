void ComputeMatrix(double* A, double* B, double* C, int dim)
{
	int i, j, k;
	for(i = 0; i < dim * dim; i++)
	{
		C[i] = 0;
		for(j = 0; j < dim; j++)
		{
			for(k = 0; k < dim; k++)
			{
				C[i] += A[j] * B[k];
			}
		}
	}
}

int main(int argc, char* argv[])
{
	int dim = 100;
	double* A = new double[dim * dim];
	double* B = new double[dim * dim];
	double* C = new double[dim * dim];

	ComputeMatrix(A, B, C, dim);
}
