#ifndef __INIDEVICE_CU__
#define __INIDEVICE_CU__
#include "gPET.h"
void printDevProp(int device)
//      print out device properties
{
    int devCount;
    cudaDeviceProp devProp;
//      device properties

    cudaGetDeviceCount(&devCount);
	printf("Number of device: %d\n", devCount);
    printf("Using device #: %d\n", device);
    cudaGetDeviceProperties(&devProp, device);
	
	printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %7.2f MB\n",  
	devProp.totalGlobalMem/1024.0/1024.0);
    printf("Total shared memory per block: %5.2f kB\n",  
	devProp.sharedMemPerBlock/1024.0);
    printf("Total registers per block:     %u\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    	
	printf("Maximum dimension of block:    %d*%d*%d\n", 			
	devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
	printf("Maximum dimension of grid:     %d*%d*%d\n", 
	devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
    printf("Clock rate:                    %4.2f GHz\n",  devProp.clockRate/1000000.0);
    printf("Total constant memory:         %5.2f kB\n",  devProp.totalConstMem/1024.0);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
//      obtain computing resource

}

void iniDevice(int deviceNo)
/*******************************************************************
c*    Initializes the physics subsystem                                  *
c******************************************************************/
{
    printf(" \n");
    printf("init: device info;\n");
    printf("information from this stream follows:\n");

    CUDA_CALL(cudaSetDevice(deviceNo));
        
    CUDA_CALL(cudaDeviceReset());
        
    printDevProp(deviceNo);

    printf("finish init: device info;\n\n");
}

#endif
