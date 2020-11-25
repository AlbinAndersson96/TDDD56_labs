
__global__ void average_precompiled_Overlap2DKernel_average_kernel_conv_cuda_2D_kernel(
	unsigned char* input, unsigned long elemPerPx,  unsigned char* output,
	const size_t out_rows, const size_t out_cols,
	size_t overlap_y, size_t overlap_x,
	size_t in_pitch, size_t out_pitch,
	const size_t sharedRows, const size_t sharedCols
)
{
   extern __shared__ unsigned char sdata[]; // will also contain extra (overlap data)
	// extern __shared__ char _sdata[];
	// unsigned char* sdata = reinterpret_cast<unsigned char*>(_sdata); // will also contain extra (overlap data)

	size_t xx = blockIdx.x * blockDim.x;
	size_t yy = blockIdx.y * blockDim.y;

	size_t x = xx + threadIdx.x;
	size_t y = yy + threadIdx.y;

	if (x < out_cols + overlap_x * 2 && y < out_rows + overlap_y * 2)
	{
		sdata[threadIdx.y * sharedCols + threadIdx.x] = input[y * in_pitch + x];

		// To load data in shared memory including neighbouring elements...
		for (size_t shared_y = threadIdx.y; shared_y < sharedRows; shared_y += blockDim.y)
		{
			for (size_t shared_x = threadIdx.x; shared_x < sharedCols; shared_x += blockDim.x)
			{
				sdata[shared_y * sharedCols + shared_x] = input[(yy + shared_y) * in_pitch + xx + shared_x];
			}
		}
	}

	__syncthreads();

	if (x < out_cols && y < out_rows)
		output[y*out_pitch+x] = skepu_userfunction_skepu_skel_1conv_average_kernel::CU({(int)overlap_x, (int)overlap_y,
			sharedCols, &sdata[(threadIdx.y + overlap_y) * sharedCols + (threadIdx.x + overlap_x)]} , elemPerPx);
}
