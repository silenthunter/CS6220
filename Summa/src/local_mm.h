#ifndef LCL_MM_
#define LCL_MM_
#ifdef _nvcc
extern "C" 
{
#endif
void local_mm (const int m, const int n, const int k, const double alpha,
	       const double *A, const int lda, const double *B, const int ldb,
	       const double beta, double *C, const int ldc);
void initDeviceProperties();
#ifdef _nvcc
}
#endif
#endif
