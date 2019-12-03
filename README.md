# CUDA-lab-for-Parallel-Computing

Created a program that creates an array of random numbers, then finds the maximum number in the array.
This job was outsourced to the CUDA GPU. The main function allocates and populates the array, then copies it over
to the GPU. The GPU divides the job among blocks to figure out the max value. The relevant info is copied back to the host. 

There is also a PDF analysis file, which compares runtimes across problem sizes.  
