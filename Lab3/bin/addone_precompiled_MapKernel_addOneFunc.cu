
__global__ void addone_precompiled_MapKernel_addOneFunc(float* skepu_output, float *a,  size_t w2, size_t w3, size_t w4, size_t n, size_t base)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	while (i < n)
	{
		
		auto res = skepu_userfunction_skepu_skel_0addOneMap_addOneFunc::CU(a[i]);
		skepu_output[i] = res;
		i += gridSize;
	}
}
