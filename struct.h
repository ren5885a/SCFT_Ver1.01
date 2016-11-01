#include <cuda_runtime.h>

#include <cufft.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <helper_functions.h>

#include <vector>
//Header stuff here
#define Pi 3.1415926535

#ifndef TAUCS_H
#define TAUCS_H




typedef struct {
	int kernal_type;
	
	int GPU_N;
	int *whichGPUs;
	cudaStream_t *stream;

	int thread; //!Maximal thread num
	int thread_sur;
	cudaDeviceProp prop[64];
	
}GPU_INFO;

typedef struct{
	

	//! scft infomation
	
	int batch; // Number of scft calculation in one GPU.
	int Num_SCFT;
	
	int intag; // Initialize type
	
	double hAB; // Flory-Huggins parameter
	double fA; 
	double fB;
	int MaxIT;//Maximum iteration steps
	int AverIt; // Output iter
	
	//! Information of the system 
	//!Grid size of the system 
	int Nx;
	int Ny;
	int Nz;
	int Nxh1;
	
	
	long NxNyNz;	//!total grid number
	long Nxh1NyNz;
	

	int NsA;
	int NsB;
	int ns;
	int dNsB;
	double dfB;
	
	double lx,ly,lz;
	double dx,dy,dz;
	double ds0,ds2; 	

	//! cufft configuration variable

	
	
	cufftHandle *plan_forward;
	cufftHandle *plan_backward;
	

	//! Temperary variables for cufft

	std::vector<double*> device_in;	//fft in in each GPU
	std::vector<cufftDoubleComplex*> device_out;// fft out in each GPU;
	

	
	
	// variable for SCFT
	
	
	double Sm1;
	double Sm2;
	
	double *kx,*kz,*ky;	
	
	double *wa;
	double *wb;
	double *wc;
	
	double *pha;
	double *phb;
	double *phc;

	

	double **kxyzdz_cu;//! pointer which is to kxyzdz in each gpu, the same in each GPU not acoording to grid.	
	double *kxyzdz;//!	kxyzdz pointer to CPU

	std::vector<double*> ql;
	std::vector<double*> ffl;

	std::vector<double*> pha_cu;	
	std::vector<double*> phb_cu;

	std::vector<double*> wa_cu;
	std::vector<double*> wb_cu;
	std::vector<double*>qInt_cu;
	std::vector<double*> wdz_cu;
	
	std::vector<double*> qa_cu;
	std::vector<double*> qb_cu;
	std::vector<double*> qca_cu;
	std::vector<double*> qcb_cu;

	

}CUFFT_INFO;
#endif //TAUCS_H

