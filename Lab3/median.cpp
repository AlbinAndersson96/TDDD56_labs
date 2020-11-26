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
//#include <array>
//#include <vector>
#include <cmath>
#include <algorithm>

#include <skepu>

#include "support.h"


unsigned char median_kernel(skepu::Region2D<unsigned char> image, size_t elemPerPx)
{
	float res = 0;
	int arrCounter = 0;
	float hold[10000];
	//std::vector<int> hold(5000); // Prevents allocation

	for (int y = -image.oi; y <= image.oi; ++y)
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
			hold[arrCounter++] = image(y, x);

    for (int i = 0; i < arrCounter; ++i) {  
		for (int j = 0; j < arrCounter-i-1; ++j) {
			if (hold[j] > hold[j+1]) {
				float temp = hold[j];  
				hold[j] = hold[j+1];  
				hold[j+1] = temp;
			}
		}
	}

	return hold[(arrCounter+1)/2];
}

int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// Skeleton instance
	auto calculateMedian = skepu::MapOverlap(median_kernel);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


