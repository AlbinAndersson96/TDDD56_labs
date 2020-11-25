#define SKEPU_PRECOMPILED
#define SKEPU_OPENMP
#define SKEPU_OPENCL
#define SKEPU_CUDA
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu>

/* SkePU user functions */

float add(float a, float b)
{
	return a+b;
}

float mult(float a, float b)
{
	return a*b;
}



struct skepu_userfunction_skepu_skel_0reduce_add
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<float, float>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{
	return a+b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{
	return a+b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{
	return a+b;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "dotproduct_precompiled_ReduceKernel_add.cu"
#include "dotproduct_precompiled_ReduceKernel_add_cl_source.inl"

struct skepu_userfunction_skepu_skel_1dotProd_mult
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{
	return a*b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{
	return a*b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{
	return a*b;
}
#undef SKEPU_USING_BACKEND_CPU
};


struct skepu_userfunction_skepu_skel_1dotProd_add
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{
	return a+b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{
	return a+b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{
	return a+b;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "dotproduct_precompiled_MapReduceKernel_mult_add.cu"
#include "dotproduct_precompiled_MapReduceKernel_mult_add_arity_2_cl_source.inl"

struct skepu_userfunction_skepu_skel_2map_mult
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float, float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a, float b)
{
	return a*b;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{
	return a*b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{
	return a*b;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "dotproduct_precompiled_MapKernel_mult.cu"
#include "dotproduct_precompiled_MapKernel_mult_arity_2_cl_source.inl"
int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
//	spec.setCPUThreads(<integer value>);
	skepu::setGlobalBackendSpec(spec);
	
	
	/* Skeleton instances */
//	auto instance = skepu::Map(userfunction);
// ...
	
	/* SkePU containers */
	skepu::Vector<float> v1(size, 1.0f), v2(size, 2.0f), v3(size, 0.0f);
	
	
	/* Compute and measure time */
	float resComb, resSep;
	
	// MapReduce
	skepu::backend::MapReduce<2, skepu_userfunction_skepu_skel_1dotProd_mult, skepu_userfunction_skepu_skel_1dotProd_add, decltype(&dotproduct_precompiled_MapReduceKernel_mult_add), decltype(&dotproduct_precompiled_MapReduceKernel_mult_add_ReduceOnly), CLWrapperClass_dotproduct_precompiled_MapReduceKernel_mult_add_arity_2> dotProd(dotproduct_precompiled_MapReduceKernel_mult_add, dotproduct_precompiled_MapReduceKernel_mult_add_ReduceOnly);
	auto timeComb = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		resComb = dotProd(v1, v2);
	});
	
	// Map + Reduce
	skepu::backend::Map<2, skepu_userfunction_skepu_skel_2map_mult, decltype(&dotproduct_precompiled_MapKernel_mult), CLWrapperClass_dotproduct_precompiled_MapKernel_mult_arity_2> map(dotproduct_precompiled_MapKernel_mult);
	skepu::backend::Reduce1D<skepu_userfunction_skepu_skel_0reduce_add, decltype(&dotproduct_precompiled_ReduceKernel_add), CLWrapperClass_dotproduct_precompiled_ReduceKernel_add> reduce(dotproduct_precompiled_ReduceKernel_add);
	auto timeSep = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		map(v3, v1, v2);
		resSep = reduce(v3);
	});
	
	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";
	
	
	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";
	
	return 0;
}

