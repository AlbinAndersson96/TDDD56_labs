/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length, __global unsigned int *intermediate, const unsigned int currentRun)
{ 
  const size_t threadIDLocal = get_local_id(0);

  const size_t threadIDGlobal1 = get_global_id(0);
  const size_t threadIDGlobal2 = threadIDGlobal1*2;
  
  // 8192 / 1024 = 8
  const size_t numberOfDigits = length / get_local_size(0);
  int i = threadIDLocal * get_local_size(0);

  int biggest = 0;
  for (; i < numberOfDigits; ++i) {
    if (biggest < data[i]) biggest = data[i];
  }

  if (intermediate[threadIDLocal*currentRun] < biggest) intermediate[threadIDLocal*currentRun] = biggest;
}