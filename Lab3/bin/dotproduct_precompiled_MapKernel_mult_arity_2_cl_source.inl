
class CLWrapperClass_dotproduct_precompiled_MapKernel_mult_arity_2
{
public:

	static cl_kernel skepu_kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
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


__kernel void dotproduct_precompiled_MapKernel_mult_arity_2( __global float* skepu_output, __global float *a, __global float *b,   size_t skepu_n, size_t skepu_base)
{
	size_t skepu_i = get_global_id(0);
	size_t skepu_gridSize = get_local_size(0) * get_num_groups(0);
	

	while (skepu_i < skepu_n)
	{
		
		
#if !0
		skepu_output[skepu_i] = mult(a[skepu_i], b[skepu_i]);
#else
		skepu_multiple skepu_out_temp = mult(a[skepu_i], b[skepu_i]);
		
#endif
		skepu_i += skepu_gridSize;
	}
}
)###");

		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu::backend::Device_CL *device : skepu::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "dotproduct_precompiled_MapKernel_mult_arity_2", &err);
			CL_CHECK_ERROR(err, "Error creating map kernel 'dotproduct_precompiled_MapKernel_mult_arity_2'");

			skepu_kernels(counter++, &kernel);
		}

		initialized = true;
	}

	template<typename Ignore>
	static void map
	(
		size_t skepu_deviceID, size_t skepu_localSize, size_t skepu_globalSize,
		 skepu::backend::DeviceMemPointer_CL<float> *skepu_output, skepu::backend::DeviceMemPointer_CL<float> *a, skepu::backend::DeviceMemPointer_CL<float> *b,  Ignore, 
		size_t skepu_n, size_t skepu_base
	)
	{
		skepu::backend::cl_helpers::setKernelArgs(skepu_kernels(skepu_deviceID),  skepu_output->getDeviceDataPointer(), a->getDeviceDataPointer(), b->getDeviceDataPointer(),   skepu_n, skepu_base);
		cl_int skepu_err = clEnqueueNDRangeKernel(skepu::backend::Environment<int>::getInstance()->m_devices_CL.at(skepu_deviceID)->getQueue(), skepu_kernels(skepu_deviceID), 1, NULL, &skepu_globalSize, &skepu_localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(skepu_err, "Error launching Map kernel");
	}
};
