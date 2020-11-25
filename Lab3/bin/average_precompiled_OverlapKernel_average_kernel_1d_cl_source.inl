
class CLWrapperClass_average_precompiled_OverlapKernel_average_kernel_1d
{
public:

	enum
	{
		KERNEL_VECTOR = 0,
		KERNEL_MATRIX_ROW,
		KERNEL_MATRIX_COL,
		KERNEL_MATRIX_COL_MULTI,
		KERNEL_COUNT
	};

	static cl_kernel kernels(size_t deviceID, size_t kerneltype, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8][KERNEL_COUNT]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID][kerneltype] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID][kerneltype];
	}

	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;

		std::string source = skepu::backend::cl_helpers::replaceSizeT(R"###(
typedef struct {
	__local unsigned char *data;
	int oi;
	size_t stride;
} skepu_region1d_unsigned__space__char;

static unsigned char skepu_region_access_1d_unsigned__space__char(skepu_region1d_unsigned__space__char r, int i)
{ return r.data[i * r.stride]; }

#define SKEPU_USING_BACKEND_CL 1

typedef struct{
	size_t i;
} index1_t;

typedef struct {
	size_t row;
	size_t col;
} index2_t;

typedef struct {
	size_t i;
	size_t j;
	size_t k;
} index3_t;

typedef struct {
	size_t i;
	size_t j;
	size_t k;
	size_t l;
} index4_t;

static size_t get_device_id()
{
	return SKEPU_INTERNAL_DEVICE_ID;
}

#define VARIANT_OPENCL(block) block
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block)

// Size of basic integer types defined in OpenCL standard.
// Emulate stdint.h based on this.
typedef uchar    uint8_t;
typedef ushort   uint16_t;
typedef uint     uint32_t;
typedef ulong    uint64_t;

typedef char     int8_t;
typedef short    int16_t;
typedef int      int32_t;
typedef long     int64_t;

enum {
	SKEPU_EDGE_NONE = 0,
	SKEPU_EDGE_CYCLIC = 1,
	SKEPU_EDGE_DUPLICATE = 2,
	SKEPU_EDGE_PAD = 3,
};

static unsigned char average_kernel_1d(skepu_region1d_unsigned__space__char m, unsigned long elemPerPx)
{
	float nrElements = m.oi*2/elemPerPx;
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += skepu_region_access_1d_unsigned__space__char(m,x);
	}
	return res/nrElements;
}


__kernel void average_precompiled_OverlapKernel_average_kernel_1d_Vector( __global unsigned char* skepu_output, __global unsigned char* m, unsigned long elemPerPx, 
	__global unsigned char* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t out_offset,
	size_t out_numelements, int poly, unsigned char pad, __local unsigned char* sdata
)
{
	size_t tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	skepu_region1d_unsigned__space__char skepu_region = { .data = &sdata[tid+skepu_overlap], .oi = skepu_overlap, .stride = 1 };

	

	if (poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap + tid] = (skepu_i < skepu_n) ? m[skepu_i] : pad;
		if (tid < skepu_overlap)
			sdata[tid] = (get_group_id(0) == 0) ? pad : m[skepu_i - skepu_overlap];

		if (tid >= get_local_size(0) - skepu_overlap)
			sdata[tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && skepu_i + skepu_overlap < skepu_n) ? m[skepu_i + skepu_overlap] : pad;
	}
	else if (poly == SKEPU_EDGE_CYCLIC)
	{
		if (skepu_i < skepu_n)
			sdata[skepu_overlap + tid] = m[skepu_i];
		else if (skepu_i - skepu_n < skepu_overlap)
			sdata[skepu_overlap + tid] = skepu_wrap[skepu_overlap + skepu_i - skepu_n];
		else
			sdata[skepu_overlap + tid] = pad;

		if (tid < skepu_overlap)
			sdata[tid] = (get_group_id(0) == 0) ? skepu_wrap[tid] : m[skepu_i - skepu_overlap];

		if (tid >= get_local_size(0) - skepu_overlap)
			sdata[tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && skepu_i + skepu_overlap < skepu_n) ? m[skepu_i + skepu_overlap] : skepu_wrap[skepu_overlap + skepu_i + skepu_overlap - skepu_n];
	}
	else if (poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[skepu_i] : m[skepu_n-1];
		if (tid < skepu_overlap)
			sdata[tid] = (get_group_id(0) == 0) ? m[0] : m[skepu_i-skepu_overlap];

		if (tid >= get_local_size(0) - skepu_overlap)
			sdata[tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && skepu_i + skepu_overlap < skepu_n) ? m[skepu_i + skepu_overlap] : m[skepu_n - 1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (skepu_i >= out_offset && skepu_i < out_offset + out_numelements)
		skepu_output[skepu_i - out_offset] = average_kernel_1d(skepu_region, elemPerPx);
}

__kernel void average_precompiled_OverlapKernel_average_kernel_1d_MatRowWise( __global unsigned char* skepu_output, __global unsigned char* m, unsigned long elemPerPx, 
	__global unsigned char* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements,
	int poly, unsigned char pad, size_t blocksPerRow, size_t rowWidth, __local unsigned char* sdata
)
{
	size_t tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex= 2 * skepu_overlap * (int)(get_group_id(0) / blocksPerRow);
	size_t tmp  = (get_group_id(0) % blocksPerRow);
	size_t tmp2 = (get_group_id(0) / blocksPerRow);
	skepu_region1d_unsigned__space__char skepu_region = { .data = &sdata[tid+skepu_overlap], .oi = skepu_overlap, .stride = 1 };

	

	if (poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[skepu_i] : pad;
		if (tid < skepu_overlap)
			sdata[tid] = (tmp==0) ? pad : m[skepu_i-skepu_overlap];

		if (tid >= (get_local_size(0)-skepu_overlap))
			sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (skepu_i+skepu_overlap < skepu_n) && tmp!=(blocksPerRow-1)) ? m[skepu_i+skepu_overlap] : pad;
	}
	else if (poly == SKEPU_EDGE_CYCLIC)
	{
		if (skepu_i < skepu_n)
			sdata[skepu_overlap+tid] = m[skepu_i];
		else if (skepu_i-skepu_n < skepu_overlap)
			sdata[skepu_overlap+tid] = skepu_wrap[(skepu_overlap+(skepu_i-skepu_n))+ wrapIndex];
		else
			sdata[skepu_overlap+tid] = pad;

		if (tid < skepu_overlap)
			sdata[tid] = (tmp==0) ? skepu_wrap[tid+wrapIndex] : m[skepu_i-skepu_overlap];

		if (tid >= (get_local_size(0)-skepu_overlap))
			sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && skepu_i+skepu_overlap < skepu_n && tmp!=(blocksPerRow-1))
				? m[skepu_i+skepu_overlap] : skepu_wrap[skepu_overlap+wrapIndex+(tid+skepu_overlap-get_local_size(0))];
	}
	else if (poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[skepu_i] : m[skepu_n-1];
		if(tid < skepu_overlap)
			sdata[tid] = (tmp==0) ? m[tmp2*rowWidth] : m[skepu_i-skepu_overlap];

		if(tid >= (get_local_size(0)-skepu_overlap))
			sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (skepu_i+skepu_overlap < skepu_n) && (tmp!=(blocksPerRow-1)))
				? m[skepu_i+skepu_overlap] : m[(tmp2+1)*rowWidth-1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if ((skepu_i >= out_offset) && (skepu_i < out_offset+out_numelements))
		skepu_output[skepu_i-out_offset] = average_kernel_1d(skepu_region, elemPerPx);
}

__kernel void average_precompiled_OverlapKernel_average_kernel_1d_MatColWise( __global unsigned char* skepu_output, __global unsigned char* m, unsigned long elemPerPx, 
	__global unsigned char* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements,
	int poly, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth, __local unsigned char* sdata
	)
{
	size_t tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex= 2 * skepu_overlap * (int)(get_group_id(0)/blocksPerCol);
	size_t tmp= (get_group_id(0) % blocksPerCol);
	size_t tmp2= (get_group_id(0) / blocksPerCol);
	size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;
	skepu_region1d_unsigned__space__char skepu_region = { .data = &sdata[tid+skepu_overlap], .oi = skepu_overlap, .stride = 1 };

	

	if (poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[arrInd] : pad;
		if (tid < skepu_overlap)
			sdata[tid] = (tmp==0) ? pad : m[(arrInd-(skepu_overlap*rowWidth))];

		if (tid >= (get_local_size(0)-skepu_overlap))
			sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(skepu_overlap*rowWidth)) < skepu_n && (tmp!=(blocksPerCol-1))) ? m[(arrInd+(skepu_overlap*rowWidth))] : pad;
	}
	else if (poly == SKEPU_EDGE_CYCLIC)
	{
		if (skepu_i < skepu_n)
			sdata[skepu_overlap+tid] = m[arrInd];
		else if (skepu_i-skepu_n < skepu_overlap)
			sdata[skepu_overlap+tid] = skepu_wrap[(skepu_overlap+(skepu_i-skepu_n))+ wrapIndex];
		else
			sdata[skepu_overlap+tid] = pad;

		if (tid < skepu_overlap)
			sdata[tid] = (tmp==0) ? skepu_wrap[tid+wrapIndex] : m[(arrInd-(skepu_overlap*rowWidth))];

		if (tid >= (get_local_size(0)-skepu_overlap))
			sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(skepu_overlap*rowWidth)) < skepu_n && (tmp!=(blocksPerCol-1)))
				? m[(arrInd+(skepu_overlap*rowWidth))] : skepu_wrap[skepu_overlap+wrapIndex+(tid+skepu_overlap-get_local_size(0))];
	}
	else if (poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[arrInd] : m[skepu_n-1];
		if (tid < skepu_overlap)
			sdata[tid] = (tmp==0) ? m[tmp2] : m[(arrInd-(skepu_overlap*rowWidth))];

		if (tid >= (get_local_size(0)-skepu_overlap))
			sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(skepu_overlap*rowWidth)) < skepu_n && (tmp!=(blocksPerCol-1)))
				? m[(arrInd+(skepu_overlap*rowWidth))] : m[tmp2+(colWidth-1)*rowWidth];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if ((arrInd >= out_offset) && (arrInd < out_offset+out_numelements))
		skepu_output[arrInd-out_offset] = average_kernel_1d(skepu_region, elemPerPx);
}

__kernel void average_precompiled_OverlapKernel_average_kernel_1d_MatColWiseMulti( __global unsigned char* skepu_output, __global unsigned char* m, unsigned long elemPerPx, 
	__global unsigned char* skepu_wrap, size_t skepu_n, size_t skepu_overlap, size_t in_offset, size_t out_numelements,
	int poly, int deviceType, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
	__local unsigned char* sdata
)
{
	size_t tid = get_local_id(0);
	size_t skepu_i   = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex = 2 * skepu_overlap * (int)(get_group_id(0)/blocksPerCol);
	size_t tmp  = (get_group_id(0) % blocksPerCol);
	size_t tmp2 = (get_group_id(0) / blocksPerCol);
	size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;
	skepu_region1d_unsigned__space__char skepu_region = { .data = &sdata[tid+skepu_overlap], .oi = skepu_overlap, .stride = 1 };

	

	if (poly == SKEPU_EDGE_PAD)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[arrInd+in_offset] : pad;
		if (deviceType == -1)
		{
			if (tid < skepu_overlap)
				sdata[tid] = (tmp==0) ? pad : m[(arrInd-(skepu_overlap*rowWidth))];

			if(tid >= (get_local_size(0)-skepu_overlap))
				sdata[tid+2*skepu_overlap] = m[(arrInd+in_offset+(skepu_overlap*rowWidth))];
		}
		else if (deviceType == 0)
		{
			if(tid < skepu_overlap)
				sdata[tid] = m[arrInd];

			if(tid >= (get_local_size(0)-skepu_overlap))
				sdata[tid+2*skepu_overlap] = m[(arrInd+in_offset+(skepu_overlap*rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (tid < skepu_overlap)
				sdata[tid] = m[arrInd];

			if (tid >= (get_local_size(0)-skepu_overlap))
				sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(skepu_overlap*rowWidth)) < skepu_n && (tmp!=(blocksPerCol-1)))
					? m[(arrInd+in_offset+(skepu_overlap*rowWidth))] : pad;
		}
	}
	else if (poly == SKEPU_EDGE_CYCLIC)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[arrInd+in_offset] : ((skepu_i-skepu_n < skepu_overlap) ? skepu_wrap[(skepu_i-skepu_n)+ (skepu_overlap * tmp2)] : pad);
		if (deviceType == -1)
		{
			if (tid < skepu_overlap)
				sdata[tid] = (tmp==0) ? skepu_wrap[tid+(skepu_overlap * tmp2)] : m[(arrInd-(skepu_overlap*rowWidth))];

			if (tid >= (get_local_size(0)-skepu_overlap))
				sdata[tid+2*skepu_overlap] = m[(arrInd+in_offset+(skepu_overlap*rowWidth))];
		}
		else if (deviceType == 0)
		{
			if (tid < skepu_overlap)
				sdata[tid] = m[arrInd];

			if (tid >= (get_local_size(0)-skepu_overlap))
				sdata[tid+2*skepu_overlap] = m[(arrInd+in_offset+(skepu_overlap*rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (tid < skepu_overlap)
				sdata[tid] = m[arrInd];

			if (tid >= (get_local_size(0)-skepu_overlap))
				sdata[tid+2*skepu_overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(skepu_overlap*rowWidth)) < skepu_n && (tmp!=(blocksPerCol-1)))
					? m[(arrInd+in_offset+(skepu_overlap*rowWidth))] : skepu_wrap[(skepu_overlap * tmp2)+(tid+skepu_overlap-get_local_size(0))];
		}
	}
	else if (poly == SKEPU_EDGE_DUPLICATE)
	{
		sdata[skepu_overlap+tid] = (skepu_i < skepu_n) ? m[arrInd + in_offset] : m[skepu_n + in_offset - 1];
		if (deviceType == -1)
		{
			if (tid < skepu_overlap)
				sdata[tid] = (tmp == 0) ? m[tmp2] : m[arrInd - skepu_overlap * rowWidth];

			if (tid >= get_local_size(0) - skepu_overlap)
				sdata[tid+2*skepu_overlap] = m[arrInd + in_offset + skepu_overlap * rowWidth];
		}
		else if (deviceType == 0)
		{
			if (tid < skepu_overlap)
				sdata[tid] = m[arrInd];

			if (tid >= get_local_size(0) - skepu_overlap)
				sdata[tid+2*skepu_overlap] = m[arrInd + in_offset + skepu_overlap * rowWidth];
		}
		else if (deviceType == 1)
		{
			if (tid < skepu_overlap)
				sdata[tid] = m[arrInd];

			if (tid >= get_local_size(0) - skepu_overlap)
				sdata[tid + 2 * skepu_overlap] = (get_group_id(0) != get_num_groups(0) - 1 && (arrInd + skepu_overlap * rowWidth < skepu_n) && (tmp != blocksPerCol - 1))
					? m[arrInd + in_offset + skepu_overlap * rowWidth] : m[tmp2 + in_offset + (colWidth - 1) * rowWidth];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (arrInd < out_numelements )
		skepu_output[arrInd] = average_kernel_1d(skepu_region, elemPerPx);
}
)###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel_vector = clCreateKernel(program, "average_precompiled_OverlapKernel_average_kernel_1d_Vector", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D vector kernel 'average_precompiled_OverlapKernel_average_kernel_1d'");

			cl_kernel kernel_matrix_row = clCreateKernel(program, "average_precompiled_OverlapKernel_average_kernel_1d_MatRowWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix row-wise kernel 'average_precompiled_OverlapKernel_average_kernel_1d'");

			cl_kernel kernel_matrix_col = clCreateKernel(program, "average_precompiled_OverlapKernel_average_kernel_1d_MatColWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise kernel 'average_precompiled_OverlapKernel_average_kernel_1d'");

			cl_kernel kernel_matrix_col_multi = clCreateKernel(program, "average_precompiled_OverlapKernel_average_kernel_1d_MatColWiseMulti", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise multi kernel 'average_precompiled_OverlapKernel_average_kernel_1d'");

			kernels(counter, KERNEL_VECTOR,           &kernel_vector);
			kernels(counter, KERNEL_MATRIX_ROW,       &kernel_matrix_row);
			kernels(counter, KERNEL_MATRIX_COL,       &kernel_matrix_col);
			kernels(counter, KERNEL_MATRIX_COL_MULTI, &kernel_matrix_col_multi);
			counter++;
		}

		initialized = true;
	}

	static void mapOverlapVector
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		 skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_output, skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_input, unsigned long elemPerPx, 
		skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements, int poly, unsigned char pad,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_VECTOR);
		skepu::backend::cl_helpers::setKernelArgs(kernel,  skepu_output->getDeviceDataPointer(), skepu_input->getDeviceDataPointer(), elemPerPx, 
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, out_offset, out_numelements, poly, pad);
		clSetKernelArg(kernel, 3 + 7, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D vector kernel");
	}

	static void mapOverlapMatrixRowWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		 skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_output, skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_input, unsigned long elemPerPx, 
		skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements, int poly, unsigned char pad, size_t blocksPerRow, size_t rowWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_ROW);
		skepu::backend::cl_helpers::setKernelArgs(kernel,  skepu_output->getDeviceDataPointer(), skepu_input->getDeviceDataPointer(), elemPerPx, 
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, out_offset, out_numelements, poly, pad, blocksPerRow, rowWidth);
		clSetKernelArg(kernel, 3 + 9, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix row-wise kernel");
	}

	static void mapOverlapMatrixColWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		 skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_output, skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_input, unsigned long elemPerPx, 
		skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t out_offset, size_t out_numelements, int poly, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL);
		skepu::backend::cl_helpers::setKernelArgs(kernel,  skepu_output->getDeviceDataPointer(), skepu_input->getDeviceDataPointer(), elemPerPx, 
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, out_offset, out_numelements, poly, pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, 3 + 10, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise kernel");
	}

	static void mapOverlapMatrixColWiseMulti
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		 skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_output, skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_input, unsigned long elemPerPx, 
		skepu::backend::DeviceMemPointer_CL<unsigned char> *skepu_wrap,
		size_t skepu_n, size_t skepu_overlap, size_t in_offset, size_t out_numelements, int poly, int deviceType, unsigned char pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL_MULTI);
		skepu::backend::cl_helpers::setKernelArgs(kernel,  skepu_output->getDeviceDataPointer(), skepu_input->getDeviceDataPointer(), elemPerPx, 
			skepu_wrap->getDeviceDataPointer(), skepu_n, skepu_overlap, in_offset, out_numelements, poly, deviceType, pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, 3 + 11, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise multi kernel");
	}
};
