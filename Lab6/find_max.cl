/*
 * Placeholder OpenCL kernel
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length)
{ 
  const size_t threadIDLocal = get_local_id(0);
  
  // 524288 / 1024 = 512
  const size_t numberOfDigits = length / get_local_size(0);

  int biggest = 0;
  unsigned int number = 0;
  for (int i = threadIDLocal; i <= (threadIDLocal+numberOfDigits); ++i) {
    number = data[i];
    if (biggest < number) biggest = number;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  if (data[threadIDLocal] < biggest) data[threadIDLocal] = biggest;
}