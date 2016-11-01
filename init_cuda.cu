
#include"struct.h"

#include "init_cuda.h"

#include "cuda_aid.cuh"

/* The number of the grid point should be divisible by GPU number*/

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}
// CUDA includes


int prime[168]={2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997};

// find the maximal factor in a integer which is smaller than 1024(Maximal thread number in cuda)
extern int factor_decompose_1024(GPU_INFO *gpu_info,long N){

	long temp;
	
	temp=N;

	int decom[10000],index=0;
	
	for(int i=0;i<168;i++){
		
		while(temp%prime[i]==0){
			temp=temp/prime[i];
			decom[index++]=prime[i];

		};
		
		

	}
	
	int elements[10000];
	//printf("index=%d\n",index);
	
	
	for(int i=0;i<index;i++) elements[i]=0;
	
	int temp_1024=1;

	for(int j=1;j<=10;j++){
		elements[j-1]=1;
  		const size_t N_t = index;
  	
  		std::vector<int> selectors(elements, elements + N_t);
	
  		
  		do{
			int combo=1;
			for (size_t i = 0; i < selectors.size(); ++i){
      				if (selectors[i]){
        				//std::cout << decom[i] << ", ";
					combo*=decom[i];
     				 }
   			}
		
			if(combo>temp_1024&&combo<=1024) temp_1024=combo;
			if(combo==1024) break;
			
  		} while (prev_permutation(selectors.begin(), selectors.end()));


	}
	
	
	return temp_1024;
	
	
}

extern void factor_decompose(GPU_INFO *gpu_info,long N, int *Nx_a,int *Ny_a,int *Nz_a){

	
	int Nx,Ny,Nz;
	long temp;
	
	temp=N;

	int decom[10000],index=0;
	for(int i=0;i<168;i++){
		
		while(temp%prime[i]==0){
			temp=temp/prime[i];
			decom[index++]=prime[i];

		};
		
		

	}
	//printf("%ld prime is ",N);
	//for(int i=0;i<index;i++) printf(" %d ",decom[i]);
	//printf("\n");

	if(temp!=1) {
			
		printf("please give a \"good\" polymer number!\n");
		exit(0);
	}

	if(index==1) {
		
		Nx=N;
		Ny=1;
		Nz=1;
		
		

	}
	else if(index==2){
		
		Nz=1;//decom[index-1]
		Ny=decom[0];
		Nx=decom[1];
		//printf("%d %d\n",Nx,Ny);
		
	}
	else if(index>2){
		
		Nx=1;
		Ny=1;
		Nz=1;
		if(index%2==0){
			
			Nz=decom[index-1]*decom[0];

			if((index-2)%4==0){

				for(int i=0;i<(index-2)/4;i++){
					Nx*=decom[i+1]*decom[index-1-i-1];
					Ny*=decom[(index-2)/4+1+i]*decom[index-1-(index-2)/4-1-i];
					
				}
				//printf("%d %d %d\n",Nx,Ny,Nz);

			}
			else if((index-2)==2){
				
				Ny=decom[1];
				Nx=decom[2];
				//printf("%d %d %d\n",Nx,Ny,Nz);

			}
			else {
				Nz*=decom[1]*decom[2];
				for(int i=0;i<(index-4)/4;i++){

					Nx*=decom[i+3]*decom[index-1-i-1];
					Ny*=decom[(index-2)/4+3+i]*decom[index-1-(index-2)/4-1-i];
				}
				//printf("%d %d %d\n",Nx,Ny,Nz);
		
			}


		}
		else{
			Nz=decom[index-1];
			if((index-1)%4==0){

				for(int i=0;i<(index-1)/4;i++){

					Nx*=decom[i]*decom[index-1-i-1];
					Ny*=decom[(index-1)/4+i]*decom[index-1-(index-1)/4-i-1];
					
				}
				//printf("%d: %d %d %d\n",index,Nx,Ny,Nz);

			}
			else if((index-1)==2){
				
				Ny=decom[0];
				Nx=decom[1];
				//printf("%d %d %d\n",Nx,Ny,Nz);

			}
			else {
				Nz*=decom[0]*decom[1];
				for(int i=0;i<(index-3)/4;i++){

					Nx*=decom[i*2+2]*decom[index-1-i*2-1];
					Ny*=decom[i*2+3]*decom[index-3-i*2];
				}
				//printf("%d %d %d\n",Nx,Ny,Nz);
		
			}


		}
		
	}
	if(N==1) {
		Nx=1;
		Ny=1;
		Nz=1;

	}
	
	if(Nx*Ny*Nz==N) {
		
		*Nx_a=Nx;
		*Ny_a=Ny;
		*Nz_a=Nz;

	}
	else {

		printf("Error Nx %d *Ny %d  *Nz %d!= N %ld\n",Nx,Ny,Nz,N);
		exit(0);
	}
}




extern void init_cuda(GPU_INFO *gpu_info,int display){
	
	int gpu_count;
	int i,j;
	cudaDeviceProp prop[64];
	int *gpuid;
	int can_access_peer_0_1;
	
	gpu_count=0;
	
	gpuid=(int*)malloc(sizeof(int));

	
	
	//checkCudaErrors(cudaGetDeviceCount(&gpu_info->GPU_N));

	//if(gpu_info->GPU_N==8) gpu_info->GPU_N=4;//! Set the number of GPU

		
	
	

	
	
	

	
	for (i=0; i < gpu_info->GPU_N; i++){
		
        	checkCudaErrors(cudaGetDeviceProperties(&gpu_info->prop[i], i));
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
		
		// Only boards based on Fermi can support P2P
		
            	gpuid[gpu_count++] = gpu_info->whichGPUs[i];
		
		if(display==1){
		
			printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, gpu_info->prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
			printf("maxThreadsDim %d %d %d\n",gpu_info->prop[i].maxThreadsDim[0],gpu_info->prop[i].maxThreadsDim[1],gpu_info->prop[i].maxThreadsDim[2]);
            		printf("maxThreadsPerBlock %d\n",gpu_info->prop[i].maxThreadsPerBlock);
			printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, prop[i].name, (IsGPUCapableP2P(&prop[i]) ? "IS " : "NOT"));
			
		}
		
		//gpu_info->thread=gpu_info->prop[i].maxThreadsPerBlock;
		
		for(j=0;j<gpu_info->GPU_N;j++){
			if(i!=j){
				
				checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpu_info->whichGPUs[i], gpu_info->whichGPUs[j]));
    				
				if(can_access_peer_0_1) {

					checkCudaErrors(cudaDeviceEnablePeerAccess(gpu_info->whichGPUs[j], 0));
				}				
				
			}
			

		}
	
        }
	

   
}

extern void initialize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){

	
	int Dim[3];
	int i,j,k;
	int rank = 3;
	int Nx=cufft_info->Nx;
	int Ny=cufft_info->Ny;
	int Nz=cufft_info->Nz;
	
	long NxNyNz=Nx*Ny*Nz,ijk;
	
	cufft_info->NxNyNz=NxNyNz;
	cufft_info->Nxh1=Nx/2+1;
	cufft_info->Nxh1NyNz=cufft_info->Nxh1*Ny*Nz;
	
	cufft_info->batch=cufft_info->Num_SCFT/gpu_info->GPU_N;  //! number of SCFT per GPU
	
	int batch=cufft_info->batch;
	
	gpu_info->thread=factor_decompose_1024(gpu_info,cufft_info->Nx*cufft_info->Ny*cufft_info->Nz);
	
	gpu_info->thread_sur=factor_decompose_1024(gpu_info,cufft_info->Nxh1*cufft_info->Ny*cufft_info->Nz);
	//printf("gpu_info->thread_sur %d\n",gpu_info->thread_sur);
	
	double dx,dy,dz;
	
	char comment[200];
	double ksq,ds0;
	ds0=cufft_info->ds0;

	cufft_info->ds2=cufft_info->ds0/2;
	cufft_info->dfB=1-cufft_info->fA;

	cufft_info->NsA = ((int)(cufft_info->fA/cufft_info->ds0+1.0e-8));
	cufft_info->dNsB = ((int)(cufft_info->dfB/cufft_info->ds0+1.0e-8));
	
	
	//!----------- Initialize GPU memery settings. ------------------------------------------------------	
	
	
	int nGPUs = gpu_info->GPU_N;
	
	cufft_info->kxyzdz_cu=(double **)malloc(sizeof(double*)*nGPUs);
	
	printf("Wonderful We have successfully initialized GPU setting.\n");

	
	//-----------! Initialize CUFFT settings. ------------------------------------------------------
	
	
	dim3 grid(cufft_info->Nx,cufft_info->Ny,cufft_info->Nz),block(1,1,1);
	
	Dim[0]=Nz;Dim[1]=Ny;Dim[2]=Nx;

	cufft_info->plan_forward=(cufftHandle *)malloc(sizeof(cufftHandle)*gpu_info->GPU_N);
	cufft_info->plan_backward=(cufftHandle *)malloc(sizeof(cufftHandle)*gpu_info->GPU_N);

	for(int gpu_index=0;gpu_index<gpu_info->GPU_N;gpu_index++){	
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[gpu_index]));
	
		checkCudaErrors(cufftCreate(&cufft_info->plan_forward[gpu_index]));
		checkCudaErrors(cufftCreate(&cufft_info->plan_backward[gpu_index]));
		
		
		if(rank==3){
			
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_forward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_D2Z, batch));
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_backward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_Z2D, batch));
			
		
		}
		else if(rank==2) {
		
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_forward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_D2Z, batch));
			checkCudaErrors(cufftPlanMany (&cufft_info->plan_backward[gpu_index], rank, Dim, NULL, 1, 1, NULL, 1, 1, CUFFT_Z2D, batch));

		}
	}
	
	
	

	
	cudaDeviceSynchronize();
	getLastCudaError("Kernel execution failed [  ]");
	printf("Wonderful We have successfully initialized cufft setting.\n");

	//-----------! Initialize malloc and initilize on CPU. ------------------------------------------------------	
	
	

	cufft_info->wa=(double*)malloc(sizeof(double)*NxNyNz*cufft_info->Num_SCFT);
	cufft_info->wb=(double*)malloc(sizeof(double)*NxNyNz*cufft_info->Num_SCFT);
	cufft_info->pha=(double*)malloc(sizeof(double)*NxNyNz*cufft_info->Num_SCFT);
	cufft_info->phb=(double*)malloc(sizeof(double)*NxNyNz*cufft_info->Num_SCFT);

	cufft_info->kx=(double *)malloc(sizeof(double)*Nx);
	cufft_info->ky=(double *)malloc(sizeof(double)*Ny);
	cufft_info->kz=(double *)malloc(sizeof(double)*Nz);

	cufft_info->dx=cufft_info->lx/(double)Nx;
	cufft_info->dy=cufft_info->ly/(double)Ny;
	cufft_info->dz=cufft_info->lz/(double)Nz;
	
	dx=cufft_info->dx;
	dy=cufft_info->dy;
	dz=cufft_info->dz;
	
	cufft_info->kxyzdz=(double *)malloc(sizeof(double)*NxNyNz);	

	for(i=0;i<=Nx/2-1;i++)cufft_info->kx[i]=2*Pi*i*1.0/Nx/dx;
	for(i=Nx/2;i<Nx;i++)cufft_info->kx[i]=2*Pi*(i-Nx)*1.0/dx/Nx;
	for(i=0;i<Nx;i++)cufft_info->kx[i]*=cufft_info->kx[i];

	for(i=0;i<=Ny/2-1;i++)cufft_info->ky[i]=2*Pi*i*1.0/Ny/dy;
	for(i=Ny/2;i<Ny;i++)cufft_info->ky[i]=2*Pi*(i-Ny)*1.0/dy/Ny;
	for(i=0;i<Ny;i++)cufft_info->ky[i]*=cufft_info->ky[i];

    	for(i=0;i<=Nz/2-1;i++)cufft_info->kz[i]=2*Pi*i*1.0/Nz/dz;
    	for(i=Nz/2;i<Nz;i++)cufft_info->kz[i]=2*Pi*(i-Nz)*1.0/dz/Nz;
    	for(i=0;i<Nz;i++)cufft_info->kz[i]*=cufft_info->kz[i];	

	
	for(k=0;k<Nz;k++)
	for(j=0;j<Ny;j++)
	for(i=0;i<Nx;i++)
	{
		ijk=(long)((k*Ny+j)*Nx+i);
		ksq=cufft_info->kx[i]+cufft_info->ky[j]+cufft_info->kz[k];
		cufft_info->kxyzdz[ijk]=exp(-ds0*ksq);
	}

	gpu_info->stream=(cudaStream_t*)malloc( sizeof(cudaStream_t)*gpu_info->GPU_N);
	
	
	FILE *fp;
	
	if(cufft_info->intag==1024){

			for(int i=0;i<cufft_info->Num_SCFT;i++){
		
				sprintf(comment,"phi_%d.dat",i+1);
	
				if((fp=fopen(comment,"r"))==false){
					printf("Configration file %s did not exist, please check it in your directory.\n",comment);
				}
			
				fgets(comment,200,fp);
				fgets(comment,200,fp);
			
				for(long ijk=0;ijk<cufft_info->NxNyNz;ijk++){
					fscanf(fp,"%lg %lg %lg %lg\n",&cufft_info->pha[ijk+i*NxNyNz],&cufft_info->phb[ijk+i*NxNyNz],&cufft_info->wa[ijk+i*NxNyNz],&cufft_info->wb[ijk+i*NxNyNz]);

				}
	
				fclose(fp);
			}

	}
	
	
	
	cufft_info->wa_cu.resize(gpu_info->GPU_N);
	cufft_info->wb_cu.resize(gpu_info->GPU_N);

	cufft_info->pha_cu.resize(gpu_info->GPU_N);
	cufft_info->phb_cu.resize(gpu_info->GPU_N);


	cufft_info->qa_cu.resize(gpu_info->GPU_N);
	cufft_info->qb_cu.resize(gpu_info->GPU_N);
	cufft_info->qca_cu.resize(gpu_info->GPU_N);
	cufft_info->qcb_cu.resize(gpu_info->GPU_N);

	cufft_info->ql.resize(gpu_info->GPU_N);
	cufft_info->ffl.resize(gpu_info->GPU_N);


	cufft_info->wdz_cu.resize(gpu_info->GPU_N);
	cufft_info->qInt_cu.resize(gpu_info->GPU_N);

	cufft_info->device_in.resize(gpu_info->GPU_N);
	cufft_info->device_out.resize(gpu_info->GPU_N);

	
	printf("Wonderful We have successfully initialized CPU setting.\n");
	
	//-----------! Initialize malloc and initilize on each GPUs. ------------------------------------------------------	

	
	
	
	
	for (i=0; i < gpu_info->GPU_N; i++){

		


		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
		checkCudaErrors(cudaStreamCreate(&gpu_info->stream[i]));
	
		checkCudaErrors(cufftSetStream(cufft_info->plan_forward[i], gpu_info->stream[i]));
		checkCudaErrors(cufftSetStream(cufft_info->plan_backward[i], gpu_info->stream[i]));
		
		checkCudaErrors(cudaMallocManaged((void**)&(cufft_info->kxyzdz_cu[i]), sizeof(double)* NxNyNz));
		checkCudaErrors(cudaMemcpy(cufft_info->kxyzdz_cu[i],  cufft_info->kxyzdz,sizeof(double)*NxNyNz,cudaMemcpyHostToDevice));
		
		
		
		
		checkCudaErrors(cudaMallocManaged(&(cufft_info->wa_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch));
		checkCudaErrors(cudaMemcpy(cufft_info->wa_cu[i],  cufft_info->wa+cufft_info->NxNyNz*batch*i,sizeof(double)*cufft_info->NxNyNz*batch,cudaMemcpyHostToDevice));
		
		checkCudaErrors(cudaMallocManaged(&(cufft_info->wb_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch));
		checkCudaErrors(cudaMemcpy(cufft_info->wb_cu[i],  cufft_info->wb+cufft_info->NxNyNz*batch*i,sizeof(double)*cufft_info->NxNyNz*batch,cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMallocManaged(&(cufft_info->pha_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch));
		checkCudaErrors(cudaMallocManaged(&(cufft_info->phb_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch));


		checkCudaErrors(cudaMallocManaged(&(cufft_info->qInt_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch));
		checkCudaErrors(cudaMallocManaged(&(cufft_info->wdz_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch));
		
		
		checkCudaErrors(cudaMallocManaged(&(cufft_info->qa_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch*(cufft_info->NsA+1)));//cufft_info->NsA (cufft_info->NsA+1)*
		checkCudaErrors(cudaMallocManaged(&(cufft_info->qca_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch*(cufft_info->NsA+1)));

		checkCudaErrors(cudaMallocManaged(&(cufft_info->qb_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch*(cufft_info->dNsB+1)));
		checkCudaErrors(cudaMallocManaged(&(cufft_info->qcb_cu[i]), sizeof(double)* cufft_info->NxNyNz*batch*(cufft_info->dNsB+1)));


		checkCudaErrors(cudaMallocManaged(&(cufft_info->ql[i]), sizeof(double)*batch));
		checkCudaErrors(cudaMallocManaged(&(cufft_info->ffl[i]), sizeof(double)*batch));

		
		checkCudaErrors(cudaMallocManaged(&(cufft_info->device_in[i]), sizeof(double)* cufft_info->NxNyNz*batch));
		checkCudaErrors(cudaMallocManaged(&(cufft_info->device_out[i]), sizeof(cufftDoubleComplex)* cufft_info->Nxh1NyNz*batch));

		checkCudaErrors(cudaDeviceSynchronize());
		
		
	}
	
	
	
	printf("Wonderful We have successfully initialized all the data.\n");
	
	
}




extern void finalize_cufft(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){
	int i,j;
	int can_access_peer_0_1;
	
	//! free memery on GPU
	

	
	for (i=0; i < gpu_info->GPU_N; i++){
		
		checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));

		checkCudaErrors(cufftDestroy(cufft_info->plan_forward[i]));
		checkCudaErrors(cufftDestroy(cufft_info->plan_backward[i]));
	
		checkCudaErrors(cudaFree(cufft_info->kxyzdz_cu[i]));
	
		checkCudaErrors(cudaFree(cufft_info->qa_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->qb_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->qca_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->qcb_cu[i]));

		checkCudaErrors(cudaFree(cufft_info->pha_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->phb_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->wa_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->wb_cu[i]));
		checkCudaErrors(cudaFree(cufft_info->wdz_cu[i]));
		
		checkCudaErrors(cudaFree(cufft_info->device_in[i]));
		checkCudaErrors(cudaFree(cufft_info->device_out[i]));


		for(j=0;j<gpu_info->GPU_N;j++){
			if(i!=j){
				checkCudaErrors(cudaSetDevice(gpu_info->whichGPUs[i]));
				checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer_0_1, i, j));
    				

				if(can_access_peer_0_1) {

					
					checkCudaErrors(cudaDeviceDisablePeerAccess(gpu_info->whichGPUs[j]));
			
					
				}// end if can_access_peer_0_1			
				
				
			}// end i!=j
			
		
		}//! end loop j
	
		cudaDeviceSynchronize();
	
	}//! end loop i
	
	
	
	//! free memery on CPU
	
	
	free(cufft_info->wa);
	free(cufft_info->wb);
	free(cufft_info->pha);
	free(cufft_info->phb);
	
	free(cufft_info->kx);
	free(cufft_info->ky);
	free(cufft_info->kz);
	
	free(gpu_info->stream);
	free(cufft_info->kxyzdz);
	free(gpu_info->whichGPUs);

	printf("Wonderful We have successfully evaculate all the memery on GPU and CPU \n");
	
	cudaDeviceReset();
}




/*
extern void test(GPU_INFO *gpu_info,CUFFT_INFO *cufft_info){
	
	int index;
	long ijk;
	
	
	long NxNyNz=cufft_info->NxNyNz;
	long Nxh1NyNz=cufft_info->Nxh1NyNz;
	
	dim3 gridDim(1,1,1),blockDim(1,1,1);
	
	for(index=0;index<gpu_info->GPU_N;index++){///gpu_info->GPU_N
		checkCudaErrors(cudaSetDevice(index));
		for(ijk=0;ijk<cufft_info->NxNyNz*cufft_info->batch;ijk++){
			if(ijk<cufft_info->NxNyNz)
			cufft_info->device_in[index][ijk]=ijk;
			else
			cufft_info->device_in[index][ijk]=ijk-NxNyNz;	
		}
		checkCudaErrors(cudaDeviceSynchronize());
		
		
		
	}

	for(index=0;index<gpu_info->GPU_N;index++){

		checkCudaErrors(cudaSetDevice(index));
		checkCudaErrors(cufftExecD2Z(cufft_info->plan_forward[index],cufft_info->device_in[index],cufft_info->device_out[index]));
	}

	for(index=0;index<gpu_info->GPU_N;index++){///gpu_info->GPU_N
		checkCudaErrors(cudaSetDevice(index));

		checkCudaErrors(cudaDeviceSynchronize());
	}
	getLastCudaError("Kernel execution failed [  ]");
	for(index=0;index<4;index++)
		//display_GPU_Complex_data<<<gridDim,blockDim>>>((cufftDoubleComplex*)cufft_info->device_out[index],index);
		for(ijk=0;ijk<10;ijk++)
		printf("%g %g\n",cufft_info->device_out[index][ijk+Nxh1NyNz].x,cufft_info->device_out[index][ijk+Nxh1NyNz].y);
	
}





extern void com_to_com1d(GPU_INFO *gpu_info,data_assem *data_test){
	cufftHandle plan;
	cufftComplex *data_in,*data_out;
	int BATCH=1;
	
	cudaMalloc((void**)&data_in,sizeof(cufftComplex)*data_test->Nx);
	checkCudaErrors(cudaMalloc((void**)&data_out,sizeof(cufftComplex)*data_test->Nx));
	
	cudaMemcpy(data_in,data_test->data_com_in,sizeof(cufftComplex)*data_test->Nx,cudaMemcpyHostToDevice);

	checkCudaErrors(cufftPlan1d(&plan,data_test->Nx,CUFFT_C2C,BATCH));
	
	cufftExecC2C(plan,data_in,data_out,CUFFT_FORWARD);

	cudaMemcpy(data_test->data_com_out,data_out,sizeof(cufftComplex)*data_test->Nx,cudaMemcpyDeviceToHost);
	printf("dd %g %g\n",data_test->data_com_out[0].x,data_test->data_com_out[0].y);
	cudaFree(data_in);
	cudaFree(data_out);
	cufftDestroy(plan);
	
}
*/
/*
extern void D1_MultipleGPU(GPU_INFO *gpu_info,data_assem *data_test,int N){
	
	
	
	cufftHandle plan_input; 
	cufftResult result;
    	result = cufftCreate(&plan_input);
	int nGPUs = 4, whichGPUs[4];
	whichGPUs[0] = 0; whichGPUs[1] = 1;whichGPUs[2] = 2;whichGPUs[3] = 3;
	
	dim3 gridDim(1,1),blockDim(10,10);
	printf("grid size on x=%d y=%d z=%d\n",gridDim.x,gridDim.y,gridDim.z);
	printf("block size on x=%d y=%d z=%d\n",blockDim.x,blockDim.y,blockDim.z);
	
	result = cufftXtSetGPUs (plan_input, nGPUs, whichGPUs);
	if(result!=CUFFT_SUCCESS){
		printf("failed to set GPU\n");
	}
	
	size_t worksize[4]; cufftComplex *host_data_input, *host_data_output; 
	int nx = 1024,ny=8,nz=8, batch = 1, rank = 3, n[3]; 
	n[0] = nx; 
	n[1]=ny;
	n[2]=nz;
	int size_of_data = sizeof(cufftComplex) * nx *ny*nz* batch;
	host_data_input = (cufftComplex*)malloc(size_of_data); 
	host_data_output = (cufftComplex*)malloc(size_of_data);
	printf("length is %d\n",nx);
	//initialize_1d_data (nx, batch, rank, n, inembed, &istride, &idist, onembed, &ostride, &odist, host_data_input, host_data_output);
		
	for(int i=0;i<nx*ny*nz;i++){
		host_data_input[i].x=i;
		host_data_input[i].y=0;
	}
	printf("finish initial\n");
	
	
	checkCufft( cufftMakePlanMany (plan_input, rank, n, NULL, 1, nx, NULL, 1, nx, CUFFT_C2C, batch, worksize));
	//result=cufftMakePlan1d(plan_input, nx, CUFFT_C2C, batch, worksize);
	
	
// cufftXtMalloc() - Malloc data on multiple GPUs 
	cudaLibXtDesc *device_data_input, *device_data_output;
	result = cufftXtMalloc (plan_input, &device_data_input, CUFFT_XT_FORMAT_INPLACE); 
	if(result!=CUFFT_SUCCESS){
		printf("failed 1\n");
	}
	
	result = cufftXtMalloc (plan_input, &device_data_output, CUFFT_XT_FORMAT_INPLACE); 

	
	printf("%zu %zu \n", device_data_input->descriptor->size[0],device_data_input->descriptor->size[1]); 		
	printf("%zu %zu \n", worksize[0],worksize[1]); 
	cudaSetDevice(0);
	//display_GPU_Complex_data<<<1,10>>>((cufftDoubleComplex*)device_data_input->descriptor->data[0]);

	if(result!=CUFFT_SUCCESS){
		printf("failed 2\n");
	}
	
	// // cufftXtMemcpy() - Copy data from host to multiple GPUs 
	result = cufftXtMemcpy (plan_input, device_data_input, host_data_input, CUFFT_COPY_HOST_TO_DEVICE); 
	
	// // cufftXtExecDescriptorC2C() - Execute FFT on multiple GPUs 
	
	//cudaSetDevice(0);
	result = cufftXtExecDescriptorC2C (plan_input, device_data_input, device_data_input, CUFFT_FORWARD); 
	
	printf("finish memcpy \n");
	
	// // cufftXtMemcpy() - Copy the data to natural order on GPUs 
	
	result = cufftXtMemcpy (plan_input, device_data_output, device_data_input, CUFFT_COPY_DEVICE_TO_DEVICE); 

	cudaSetDevice(0);
	//display_GPU_Complex_data<<<gridDim,blockDim>>>((cufftComplex*)device_data_output->descriptor->data[0],N);
	cudaDeviceSynchronize();
	if(result!=CUFFT_SUCCESS){
		printf("failed copy data from device to device\n");
	}
	printf("problem 1\n");
// // cufftXtMemcpy() - Copy natural order data from multiple GPUs to host 
	result = cufftXtMemcpy (plan_input, host_data_output, device_data_input, CUFFT_COPY_DEVICE_TO_HOST); 

	for(int i=0;i<8;i++){
		printf("%g %g\n",host_data_output[i].x,host_data_output[i].y);
	}

	// // Print output and check results int output_return = output_1d_results (nx, batch, host_data_input, host_data_output); // 
	// cufftXtFree() - Free GPU memory 

	
	result = cufftXtFree(device_data_input); 
	result = cufftXtFree(device_data_output); 
	// // cufftDestroy() - Destroy FFT plan 
	result = cufftDestroy(plan_input); 
	free(host_data_input); 
	free(host_data_output);

	



	

}


*/


/*
FILE *dp;
	double *kxkykz,testD;
	kxkykz=(double *)malloc(sizeof(double)*cufft_info->NxNyNz);
dp=fopen("kxyzdz.dat","r");
	testD=0;
	for(ijk=0;ijk<cufft_info->NxNyNz;ijk++){
		fscanf(dp,"%lg\n",&kxkykz[ijk]);
		testD+=(kxkykz[ijk]-cufft_info->kxyzdz[ijk])*(kxkykz[ijk]-cufft_info->kxyzdz[ijk]);
	}
	printf("compare %g\n",testD);
*/













