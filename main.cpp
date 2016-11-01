#include <stdio.h>

#include <assert.h>

#include<stdlib.h>

#include "struct.h"

#include "init_cuda.h"

//#include "/usr/include/fftw3.h"
#include "init.h"

#include "cuda_scft.h"

#include <ctime>
// z dimension must be larger than Nz/GPU_N>=8
// CUDA runtime

int main(int argc, char **argv){

	long NxNyNz,ijk;
	
	GPU_INFO gpu_info;

	CUFFT_INFO cufft_info;
	
	init_scft(&cufft_info,&gpu_info,argc, argv);

	
	fft_test(&gpu_info,&cufft_info);


	if(gpu_info.kernal_type==1){

		
		finalize_cufft(&gpu_info,&cufft_info);
	}
	
	return 0;
}
