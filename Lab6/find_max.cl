/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length, __global unsigned int *intermediate, const unsigned int currentRun)
{ 
  const size_t threadIDLocal = get_local_id(0);

  const size_t threadIDGlobal1 = get_global_id(0);
  const size_t threadIDGlobal2 = threadIDGlobal1*2;
  
  // 1048576 / 1024 = 1024
  const size_t numberOfDigits = length / get_local_size(0);
  int i = threadIDLocal * get_local_size(0);

  int biggest = 0;
  unsigned int number = 0;
  for (; i < (threadIDLocal+1) * get_local_size(0); ++i) {
    number = data[i];
    if (biggest < number) biggest = number;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (data[threadIDLocal*currentRun] < biggest) data[threadIDLocal*currentRun] = biggest;
}