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

__kernel void bitonic(__global unsigned int *data, const unsigned int N, const unsigned int j)
{ 

  // 1 2 - 1 5 - 1 1

  // 4 7 3 2 9 8 1 5
  // 4 7 - 3 2 - 9 8 - 1 5
  // 3 2 4 7 - 1 5 9 8
  // 1 2 4 7 3 5 9 8

  //unsigned int i,j,k;

  //printf("GPU sorting.\n");

  // i = get_local_id(0);
  // if (i < N) {
  //   for (k=2;k<=N;k=2*k) // Outer loop, double size for each step
  //   {
  //     for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
  //     {
  //         int ixj=i^j; // Calculate indexing!
  //         if ((ixj)>i)
  //         {
  //           if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
  //           if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
  //         }
  //     }
  //   }
  // }
  unsigned int i = get_local_id(0);
  int ixj=i^j; 
  // Calculate indexing!
  if ((ixj)>i) {
    if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
    if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
  }

}
