
__global__ void dotproduct_precompiled_ReduceKernel_add(float *input, float *output, size_t n, size_t blockSize, bool nIsPow2)
{
	extern __shared__ float sdata[];
	// extern __shared__ alignas(float) char _sdata[];
	// float *sdata = reinterpret_cast<float*>(_sdata);

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t tid = threadIdx.x;
	size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
	size_t gridSize = blockSize * 2 * gridDim.x;

	float result;

	if (i < n)
	{
		result = input[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		//This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
		// dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
		// There we pass it always false
		if (nIsPow2 || i + blockSize < n)
			result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, input[i+blockSize]);
		i += gridSize;
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, input[i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, input[i+blockSize]);
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = result;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, sdata[tid +  64]); } __syncthreads(); }

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float* smem = sdata;
		if (blockSize >=  64) { smem[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, smem[tid + 32]); }
		if (blockSize >=  32) { smem[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, smem[tid + 16]); }
		if (blockSize >=  16) { smem[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, smem[tid +  8]); }
		if (blockSize >=   8) { smem[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, smem[tid +  4]); }
		if (blockSize >=   4) { smem[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, smem[tid +  2]); }
		if (blockSize >=   2) { smem[tid] = result = skepu_userfunction_skepu_skel_0reduce_add::CU(result, smem[tid +  1]); }
	}

	// write result for this block to global mem
	if (tid == 0)
		output[blockIdx.x] = sdata[0];
}
