/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length, __global unsigned int *intermediate, const unsigned int currentRun)
{ 
  const size_t threadIDLocal = get_local_id(0);

  const size_t threadIDGlobal1 = get_global_id(0);
  const size_t threadIDGlobal2 = threadIDGlobal1*2;
  
  // 524288 / 1024 = 512
  const size_t numberOfDigits = length / get_local_size(0);
  int i = threadIDLocal;

  int biggest = 0;
  unsigned int number = 0;
  for (; i < (threadIDLocal+numberOfDigits); ++i) {
    number = data[i];
    if (biggest < number) biggest = number;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (data[threadIDLocal] < biggest) data[threadIDLocal] = biggest;
}