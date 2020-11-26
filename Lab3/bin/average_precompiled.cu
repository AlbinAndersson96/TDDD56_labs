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

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu>

#include "support.h"

unsigned char average_kernel(skepu::Region2D<unsigned char> m, size_t elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));	
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}

unsigned char average_kernel_1d(skepu::Region1D<unsigned char> m, size_t elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x);
	}
	return res * scaling;
}


unsigned char gaussian_kernel(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, size_t elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x) * stencil(x);
	}
	return res * scaling;
}



struct skepu_userfunction_skepu_skel_0conv_gaussian_kernel
{
constexpr static size_t totalArity = 3;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<const skepu::Vec<float>>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<skepu::ProxyTag::Default> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
skepu::AccessMode::Read, };

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ unsigned char CU(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x) * stencil(x);
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x) * stencil(x);
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region1D<unsigned char> m, const skepu::Vec<float> stencil, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x) * stencil(x);
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "average_precompiled_Overlap1DKernel_gaussian_kernel.cu"
#include "average_precompiled_OverlapKernel_gaussian_kernel_cl_source.inl"

struct skepu_userfunction_skepu_skel_1conv_average_kernel
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ unsigned char CU(skepu::Region2D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));	
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region2D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));	
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region2D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / ((m.oj/elemPerPx*2+1)*(m.oi*2+1));	
	float res = 0;
	for (int y = -m.oi; y <= m.oi; ++y)
		for (int x = -m.oj; x <= m.oj; x += elemPerPx)
			res += m(y, x);
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "average_precompiled_Overlap2DKernel_average_kernel.cu"
#include "average_precompiled_Overlap2DKernel_average_kernel_cl_source.inl"

struct skepu_userfunction_skepu_skel_2conv_average_kernel_1d
{
constexpr static size_t totalArity = 2;
constexpr static size_t outArity = 1;
constexpr static bool indexed = 0;
using IndexType = void;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<unsigned long>;
typedef std::tuple<> ProxyTags;
constexpr static skepu::AccessMode anyAccessMode[] = {
};

using Ret = unsigned char;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_CUDA 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE __device__ unsigned char CU(skepu::Region1D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x);
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CUDA

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char OMP(skepu::Region1D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x);
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE unsigned char CPU(skepu::Region1D<unsigned char> m, unsigned long elemPerPx)
{
	float scaling = 1.0 / (m.oi/elemPerPx*2+1);
	float res = 0;
	for (int x = -m.oi; x <= m.oi; x += elemPerPx) {
		res += m(x);
	}
	return res * scaling;
}
#undef SKEPU_USING_BACKEND_CPU
};

#include "average_precompiled_Overlap1DKernel_average_kernel_1d.cu"
#include "average_precompiled_OverlapKernel_average_kernel_1d_cl_source.inl"
int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	LodePNGColorType colorType = LCT_RGB;
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFile = outputFileName + ss.str();
	
	// Read the padded image into a matrix. Create the output matrix without padding.
	// Padded version for 2D MapOverlap, non-padded for 1D MapOverlap
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrixPad = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> inputMatrix = ReadPngFileToMatrix(inputFileName, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	skepu::Matrix<unsigned char> outputMatrixAvg(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	skepu::Matrix<unsigned char> outputMatrixGaus(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	// more containers...?
	
	// Original version
	{
		skepu::backend::MapOverlap2D<skepu_userfunction_skepu_skel_1conv_average_kernel, decltype(&average_precompiled_Overlap2DKernel_average_kernel_conv_cuda_2D_kernel), CLWrapperClass_average_precompiled_Overlap2DKernel_average_kernel> conv(average_precompiled_Overlap2DKernel_average_kernel_conv_cuda_2D_kernel);
		conv.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv(outputMatrix, inputMatrixPad, imageInfo.elementsPerPixel);
		});
	
		WritePngFileMatrix(outputMatrix, outputFile + "-average.png", colorType, imageInfo);
		std::cout << "Time for combined: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable version
	// use conv.setOverlapMode(skepu::Overlap::[ColWise RowWise]);
	// and conv.setOverlap(<integer>)
	{
		skepu::backend::MapOverlap1D<skepu_userfunction_skepu_skel_2conv_average_kernel_1d, decltype(&average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU), decltype(&average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_Row), decltype(&average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_Col), decltype(&average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_ColMulti), CLWrapperClass_average_precompiled_OverlapKernel_average_kernel_1d> conv(average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU, average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_Row, average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_Col, average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_ColMulti);
		conv.setOverlapMode(skepu::Overlap::ColRowWise);
		conv.setOverlap(radius  * imageInfo.elementsPerPixel);
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv(outputMatrixAvg, inputMatrix, imageInfo.elementsPerPixel);
		});
		
		WritePngFileMatrix(outputMatrixAvg, outputFile + "-separable.png", colorType, imageInfo);
		std::cout << "Time for separable: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	// Separable gaussian
	{
		skepu::Vector<float> stencil = sampleGaussian(radius);
			
		// skeleton instance, etc here (remember to set backend)
		skepu::backend::MapOverlap1D<skepu_userfunction_skepu_skel_0conv_gaussian_kernel, decltype(&average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU), decltype(&average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU_Matrix_Row), decltype(&average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU_Matrix_Col), decltype(&average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU_Matrix_ColMulti), CLWrapperClass_average_precompiled_OverlapKernel_gaussian_kernel> conv(average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU, average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU_Matrix_Row, average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU_Matrix_Col, average_precompiled_Overlap1DKernel_gaussian_kernel_MapOverlapKernel_CU_Matrix_ColMulti);
		conv.setOverlapMode(skepu::Overlap::ColRowWise);
		conv.setOverlap(radius  * imageInfo.elementsPerPixel);
	
		auto timeTaken = skepu::benchmark::measureExecTime([&]
		{
			conv(outputMatrixGaus, inputMatrix, stencil, imageInfo.elementsPerPixel);
		});
	
		WritePngFileMatrix(outputMatrixGaus, outputFile + "-gaussian.png", colorType, imageInfo);
		std::cout << "Time for gaussian: " << (timeTaken.count() / 10E6) << "\n";
	}
	
	
	
	return 0;
}


