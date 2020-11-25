
class CLWrapperClass_dotproduct_precompiled_MapReduceKernel_mult_add_arity_2
{
public:

	enum
	{
		KERNEL_MAPREDUCE = 0,
		KERNEL_REDUCE,
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

static float mult(float a, float b)
{
	return a*b;
}

static float add(float a, float b)
{
	return a+b;
}


__kernel void dotproduct_precompiled_MapReduceKernel_mult_add_arity_2(__global float * user_a, __global float * user_b,  __global float* skepu_output,  size_t skepu_n, size_t skepu_base, __local float* skepu_sdata)
{
	size_t skepu_blockSize = get_local_size(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * skepu_blockSize + skepu_tid;
	size_t skepu_gridSize = skepu_blockSize * get_num_groups(0);
	float skepu_result;
	

	if (skepu_i < skepu_n)
	{
		
		
		skepu_result = mult(user_a[skepu_i], user_b[skepu_i]);
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		
		
		float tempMap = mult(user_a[skepu_i], user_b[skepu_i]);
		skepu_result = add(skepu_result, tempMap);
		skepu_i += skepu_gridSize;
	}

	skepu_sdata[skepu_tid] = skepu_result;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }

	if (skepu_tid == 0)
	{
		skepu_output[get_group_id(0)] = skepu_sdata[skepu_tid];
	}
}

__kernel void dotproduct_precompiled_MapReduceKernel_mult_add_arity_2_ReduceOnly(__global float* skepu_input, __global float* skepu_output, size_t skepu_n, __local float* skepu_sdata)
{
	size_t skepu_blockSize = get_local_size(0);
	size_t skepu_tid = get_local_id(0);
	size_t skepu_i = get_group_id(0) * skepu_blockSize + get_local_id(0);
	size_t skepu_gridSize = skepu_blockSize * get_num_groups(0);
	float skepu_result;

	if (skepu_i < skepu_n)
	{
		skepu_result = skepu_input[skepu_i];
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		skepu_result = add(skepu_result, skepu_input[skepu_i]);
		skepu_i += skepu_gridSize;
	}

	skepu_sdata[skepu_tid] = skepu_result;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 512]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { skepu_sdata[skepu_tid] = add(skepu_sdata[skepu_tid], skepu_sdata[skepu_tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }

	if (skepu_tid == 0)
	{
		skepu_output[get_group_id(0)] = skepu_sdata[skepu_tid];
	}
}
)###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel_mapreduce = clCreateKernel(program, "dotproduct_precompiled_MapReduceKernel_mult_add_arity_2", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce kernel 'dotproduct_precompiled_MapReduceKernel_mult_add_arity_2'");

			cl_kernel kernel_reduce = clCreateKernel(program, "dotproduct_precompiled_MapReduceKernel_mult_add_arity_2_ReduceOnly", &err);
			CL_CHECK_ERROR(err, "Error creating MapReduce kernel 'dotproduct_precompiled_MapReduceKernel_mult_add_arity_2'");

			kernels(counter, KERNEL_MAPREDUCE, &kernel_mapreduce);
			kernels(counter, KERNEL_REDUCE,    &kernel_reduce);
			counter++;
		}

		initialized = true;
	}

	template<typename Ignore>
	static void mapReduce
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		skepu::backend::DeviceMemPointer_CL<const float> * user_a, skepu::backend::DeviceMemPointer_CL<const float> * user_b, 
		skepu::backend::DeviceMemPointer_CL<float> *skepu_output,
		Ignore,  size_t skepu_n, size_t skepu_base,
		size_t skepu_sharedMemSize
	)
	{
		cl_kernel skepu_kernel = kernels(skepu_deviceID, KERNEL_MAPREDUCE);
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernel, user_a->getDeviceDataPointer(), user_b->getDeviceDataPointer(),  skepu_output->getDeviceDataPointer(),  skepu_n, skepu_base);
		clSetKernelArg(skepu_kernel, 5, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernel, 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapReduce kernel");
	}

	static void reduceOnly
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		skepu::backend::DeviceMemPointer_CL<float> *skepu_input, skepu::backend::DeviceMemPointer_CL<float> *skepu_output,
		size_t skepu_n, size_t skepu_sharedMemSize
	)
	{
		cl_kernel skepu_kernel = kernels(skepu_deviceID, KERNEL_REDUCE);
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernel, skepu_input->getDeviceDataPointer(), skepu_output->getDeviceDataPointer(), skepu_n);
		clSetKernelArg(skepu_kernel, 3, skepu_sharedMemSize, NULL);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernel, 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching MapReduce reduce-only kernel");
	}
};
