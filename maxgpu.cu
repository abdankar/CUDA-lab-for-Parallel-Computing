#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h> 
#include <string.h>

// Parallel Computing Lab 3
// Abhi Dankar
// Professor Zahran
// All experiments run on cuda1

unsigned int getmax(unsigned int *, unsigned int);
unsigned int getmaxSEQ(unsigned int *, unsigned int);
void printArr(unsigned int pr[], unsigned int box);

#define THREADS_PER_BLOCK 1024

__global__ void getMaxCu(unsigned int* arrIn, unsigned int size, unsigned int* res) {
  unsigned int i = 0;
  __shared__ unsigned int arr[THREADS_PER_BLOCK];
  i =  threadIdx.x + (blockIdx.x * blockDim.x);
  unsigned int gb = gridDim.x*blockDim.x;
  unsigned int o = 0;

  //find max within block
  unsigned int temp = 0;
  while(i + o < size){
    temp = max(temp, arrIn[i + o]);
    o += gb;
  }
  arr[threadIdx.x] = temp;
  __syncthreads();

  //across blocks
  for (unsigned int y = (blockDim.x/2); y > 0; y = y/2){
      if ((threadIdx.x < y)){
        arr[threadIdx.x] = max(arr[threadIdx.x], arr[threadIdx.x + y]);
      }
      __syncthreads();
  }
  //get the max to return from 0 thread
  if (threadIdx.x == 0)
    res[blockIdx.x] = max(res[blockIdx.x],arr[0]);
}



int main(int argc, char *argv[]) {
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array
    
    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }
   
    size = atol(argv[1]);

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }    

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1 
    for( i = 0; i < size; i++)
       numbers[i] = rand()  % size;  

    printf(" The maximum number in the array is: %u\n", 
           getmax(numbers, size));

    free(numbers);
    exit(0);
}

void printArr(unsigned int pr[], unsigned int box){
  for (unsigned int i = 0; i < box; i++){
    printf("%u ", pr[i]);
  } 
}

//sequential getMax for checking
unsigned int getmaxSEQ(unsigned int num[], unsigned int size) {
  unsigned int i;
  unsigned int max = num[0];
  for(i = 1; i < size; i++){
    if(num[i] > max){
      max = num[i];
   }
  }
  return( max );
}
/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmax(unsigned int num[], unsigned int size) {
  unsigned int i;
  unsigned int * copy;
  unsigned int * ans; 
  unsigned int * output;
  unsigned int * newArr;
  unsigned int resize;
  //in case array doesn't fill up the block
  if (size % THREADS_PER_BLOCK != 0){
    resize = (size/THREADS_PER_BLOCK+1)*THREADS_PER_BLOCK;
  } else {
    resize = size;
  }
  //create new array with 0 values for unfilled block
  newArr = (unsigned int *) malloc(sizeof(unsigned int) * resize);
  for (i = 0; i < resize; i++){
    if (i < size){
      newArr[i] = num[i];
    } else {
      newArr[i] = 0;
    }
  }

  //how many blocks
  unsigned int blocks = (resize/THREADS_PER_BLOCK);
  cudaMalloc((void **) &copy, sizeof(unsigned int)*resize);
  cudaMemcpy((void *)copy, (void *) newArr, resize*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMalloc((void **) &output, sizeof(unsigned int)*blocks);
  ans = (unsigned int*) malloc(sizeof(unsigned int) * blocks);

  do {
    blocks = ceil((float)(resize)/(float)THREADS_PER_BLOCK);
    getMaxCu<<<blocks, THREADS_PER_BLOCK>>>(copy, resize, output);
    resize = blocks;
    copy = output;
  } while (blocks > 1);

  cudaMemcpy((void *)ans, (void *)output, blocks * sizeof(unsigned int),cudaMemcpyDeviceToHost);
  unsigned int ret = ans[0];
  cudaFree(output);
  cudaFree(copy);
  free(newArr);
  free(ans);
  return(ret);
}


