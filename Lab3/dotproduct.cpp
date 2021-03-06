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
	auto dotProd = skepu::MapReduce<2>(mult, add);
	auto timeComb = skepu::benchmark::measureExecTimeIdempotent([&]
	{
		resComb = dotProd(v1, v2);
	});
	
	// Map + Reduce
	auto map = skepu::Map<2>(mult);
	auto reduce = skepu::Reduce(add);
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