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

float addOneFunc(float a)
{
	return a+1;
}



struct skepu_userfunction_skepu_skel_0addOneMap_addOneFunc
{
constexpr static size_t totalArity = 1;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<float>;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ float CU(float a)
{
	return a+1;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a)
{
	return a+1;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a)
{
	return a+1;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "addone_precompiled_MapKernel_addOneFunc.cu"
#include "addone_precompiled_MapKernel_addOneFunc_arity_1_cl_source.inl"
int main(int argc, const char* argv[])
{
	/* Program parameters */
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	auto spec = skepu::BackendSpec{argv[2]};
	skepu::setGlobalBackendSpec(spec);
	
	/* Skeleton instances */
	skepu::backend::Map<1, skepu_userfunction_skepu_skel_0addOneMap_addOneFunc, decltype(&addone_precompiled_MapKernel_addOneFunc), CLWrapperClass_addone_precompiled_MapKernel_addOneFunc_arity_1> addOneMap(addone_precompiled_MapKernel_addOneFunc);
	
	/* SkePU containers */
	skepu::Vector<float> input(size), res(size);
//	input.randomize(0, 9);
	
	
	// This is how to measure execution times with SkePU
	auto dur = skepu::benchmark::measureExecTime([&]
	{
		// Code to be measured here
		addOneMap(res, input);
	});
	
	/* This is how to print the time */
	std::cout << "Time: " << (dur.count() / 10E6) << " seconds.\n";
	
	
	/* Print vector for debugging */
//	std::cout << "Input:  " << input << "\n";
//	std::cout << "Result: " << res << "\n";
	
	
	return 0;
}

