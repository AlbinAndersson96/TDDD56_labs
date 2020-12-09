/*
 * Placeholder OpenCL kernel
 */

void exchange(__global unsigned int *i, __global unsigned int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

__kernel void bitonic(__global unsigned int *data, const unsigned int N, const unsigned int j, const unsigned int k)
{ 

  unsigned int i = get_global_id(0);
  int ixj=i^j; 
  // Calculate indexing!
  if ((ixj)>i) {
    if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
    if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
  }

}
