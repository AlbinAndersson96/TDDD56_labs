
__global__ void dotproduct_precompiled_MapKernel_mult(float* skepu_output, float *a, float *b,  size_t w2, size_t w3, size_t w4, size_t n, size_t base)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	while (i < n)
	{
		
		auto res = skepu_userfunction_skepu_skel_2map_mult::CU(a[i], b[i]);
		skepu_output[i] = res;
		i += gridSize;
	}
}
