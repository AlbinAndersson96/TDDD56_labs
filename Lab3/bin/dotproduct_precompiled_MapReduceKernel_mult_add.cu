
__global__ void dotproduct_precompiled_MapReduceKernel_mult_add(float *a, float *b,  float *output,  size_t w2, size_t w3, size_t w4, size_t n, size_t base)
{
	extern __shared__ float sdata[];
	// extern __shared__ alignas(float) char _sdata[];
	// float *sdata = reinterpret_cast<float*>(_sdata);

	size_t blockSize = blockDim.x;
	size_t tid = threadIdx.x;
	size_t i = blockIdx.x * blockSize + tid;
	size_t gridSize = blockSize * gridDim.x;
	float result;

	if (i < n)
	{
		
		result = skepu_userfunction_skepu_skel_1dotProd_mult::CU(a[i], b[i]);
		i += gridSize;
	}

	while (i < n)
	{
		
		float tempMap = skepu_userfunction_skepu_skel_1dotProd_mult::CU(a[i], b[i]);
		result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, tempMap);
		i += gridSize;
	}

	sdata[tid] = result;
	__syncthreads();

	if (blockSize >= 1024) { if (tid < 512 && tid + 512 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >=  512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >=  256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >=  128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid +  64]); } __syncthreads(); }
	if (blockSize >=   64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid +  32]); } __syncthreads(); }
	if (blockSize >=   32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid +  16]); } __syncthreads(); }
	if (blockSize >=   16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid +   8]); } __syncthreads(); }
	if (blockSize >=    8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid +   4]); } __syncthreads(); }
	if (blockSize >=    4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid +   2]); } __syncthreads(); }
	if (blockSize >=    2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = skepu_userfunction_skepu_skel_1dotProd_add::CU(sdata[tid], sdata[tid +   1]); } __syncthreads(); }

	if (tid == 0)
		output[blockIdx.x] = sdata[tid];
}

__global__ void dotproduct_precompiled_MapReduceKernel_mult_add_ReduceOnly(float *input, float *output, size_t n, size_t blockSize, bool nIsPow2)
{
	extern __shared__ float sdata[];
	// extern __shared__ alignas(float) char _sdata[];
	// float* sdata = reinterpret_cast<float*>(_sdata);

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t tid = threadIdx.x;
	size_t i = blockIdx.x*blockSize*2 + threadIdx.x;
	size_t gridSize = blockSize*2*gridDim.x;
	float result;

	if(i < n)
	{
		result = input[i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		//This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
		// dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
		// There we pass it always false
		if (nIsPow2 || i + blockSize < n)
			result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, input[i+blockSize]);
		i += gridSize;
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a lParamer gridSize and therefore fewer elements per thread
	while(i < n)
	{
		result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, input[i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || i + blockSize < n)
			result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, input[i+blockSize]);
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = result;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, sdata[tid + 512]); } __syncthreads(); }
	if (blockSize >=  512) { if (tid < 256) { sdata[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >=  256) { if (tid < 128) { sdata[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >=  128) { if (tid <  64) { sdata[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, sdata[tid +  64]); } __syncthreads(); }

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float* smem = sdata;
		if (blockSize >=  64) { smem[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, smem[tid + 32]); }
		if (blockSize >=  32) { smem[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, smem[tid + 16]); }
		if (blockSize >=  16) { smem[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, smem[tid +  8]); }
		if (blockSize >=   8) { smem[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, smem[tid +  4]); }
		if (blockSize >=   4) { smem[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, smem[tid +  2]); }
		if (blockSize >=   2) { smem[tid] = result = skepu_userfunction_skepu_skel_1dotProd_add::CU(result, smem[tid +  1]); }
	}

	// write result for this block to global mem
	if (tid == 0)
		output[blockIdx.x] = sdata[0];
}
