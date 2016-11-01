#include "cuda_aid.cuh"

__global__ void w_to_phi(double *phlA, double *phlB,double *qA_cu,double *qcA_cu,double *qB_cu,double *qcB_cu,int NsA,int dNsB,double *ffl){
	
	
	
	
	
	long NxNyNz=gridDim.x*gridDim.y*gridDim.z;
	
	long i=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;

	long offset=threadIdx.x*NxNyNz;

	long offset_qA=threadIdx.x*NxNyNz*(NsA+1);

	long offset_qB=threadIdx.x*NxNyNz*(dNsB+1);
	
	int iz;

	long ijkiz;
	
	phlA[i+offset]=0.0;
	phlB[i+offset]=0.0;
	
	
	
	
	for(iz=0;iz<=NsA;iz++){
		
		ijkiz=i*(NsA+1)+iz+offset_qA;

		if(iz==0||(iz==NsA))phlA[i+offset]+=(0.50*qA_cu[ijkiz]*qcA_cu[ijkiz]);

		else phlA[i+offset]+=qA_cu[ijkiz]*qcA_cu[ijkiz];
			
	
	}
	
	
	
	for(iz=0;iz<=dNsB;iz++){

		ijkiz=i*(dNsB+1)+iz+offset_qB;

		if(iz==0||(iz==dNsB))phlB[i+offset]+=(0.50*qB_cu[ijkiz]*qcB_cu[ijkiz]);
		else phlB[i+offset]+=qB_cu[ijkiz]*qcB_cu[ijkiz];
		
	}
	

	
		phlA[i+offset]*=ffl[threadIdx.x];
		phlB[i+offset]*=ffl[threadIdx.x];
	

__syncthreads();
	

}
__global__ void w_to_phi_go(double *phlA, double *phlB,double *qA_cu,double *qcA_cu,double *qB_cu,double *qcB_cu,int NsA,int dNsB,double *ffl){
	
	
	
	
	
	long NxNyNz=blockDim.x*gridDim.x;
	
	long i=threadIdx.x+blockDim.x*blockIdx.x;

	long offset=blockIdx.y*NxNyNz;

	long offset_qA=blockIdx.y*NxNyNz*(NsA+1);

	long offset_qB=blockIdx.y*NxNyNz*(dNsB+1);
	
	int iz;

	long ijkiz;
	
	phlA[i+offset]=0.0;
	phlB[i+offset]=0.0;
	
	
	
	
	for(iz=0;iz<=NsA;iz++){
		
		ijkiz=i*(NsA+1)+iz+offset_qA;

		if(iz==0||(iz==NsA))phlA[i+offset]+=(0.50*qA_cu[ijkiz]*qcA_cu[ijkiz]);

		else phlA[i+offset]+=qA_cu[ijkiz]*qcA_cu[ijkiz];
			
	
	}
	
	
	
	for(iz=0;iz<=dNsB;iz++){

		ijkiz=i*(dNsB+1)+iz+offset_qB;

		if(iz==0||(iz==dNsB))phlB[i+offset]+=(0.50*qB_cu[ijkiz]*qcB_cu[ijkiz]);
		else phlB[i+offset]+=qB_cu[ijkiz]*qcB_cu[ijkiz];
		
	}
	

	
		phlA[i+offset]*=ffl[blockIdx.y];
		phlB[i+offset]*=ffl[blockIdx.y];
	

__syncthreads();
	

}
__global__ void cal_ql(double *ql_cu,double *qB_cu,int dNsB,int NxNyNz){
	
	
	
	

	extern __shared__ double sum[];
	
	double temp;
	double mySum;

	int tid=threadIdx.x;

	long offset_qB=NxNyNz*(dNsB+1)*blockIdx.x;
	
	 mySum=0;
	
	for (unsigned int s=0; s<(NxNyNz/blockDim.x+1); s++) {
		
		temp=((s+threadIdx.x*(NxNyNz/blockDim.x+1))<NxNyNz)?qB_cu[(s+threadIdx.x*(NxNyNz/blockDim.x+1))*(dNsB+1)+dNsB+offset_qB] : 0;
		
		mySum = mySum + temp;
    

		__syncthreads();
    }
    	sum[threadIdx.x] = mySum;	
	
	//if(threadIdx.x%1024==1024) printf("%g %g\n",sum[0],sum[1]);//blockDim.x
 	
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
		 
            sum[tid] += sum[tid + s];
        }

        __syncthreads();
    }
		

	
	

if(threadIdx.x==0)

	ql_cu[blockIdx.x]=sum[0];

}

__global__ void qa_to_qInt(double *qInt,double *qA,int NsA){

	long NxNyNz=gridDim.x*gridDim.y*gridDim.z;
	
	long i=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long offset=threadIdx.x*NxNyNz;

	qInt[i+offset]=qA[i*(NsA+1)+NsA+offset*(NsA+1)];
	
	__syncthreads();

}
__global__ void qa_to_qInt2(double *qInt,double *qA,int NsA){

	long NxNyNz=gridDim.x*gridDim.y*gridDim.z;
	
	long i=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long offset=threadIdx.x*NxNyNz;

	qInt[i+offset]=qA[i*(NsA+1)+offset*(NsA+1)];
	
	__syncthreads();
}


__global__ void qInt_init(double *qInt){
	
	long i=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;

	long NxNyNz=gridDim.x*gridDim.y*gridDim.z;

	qInt[i+threadIdx.x*NxNyNz]=1.0;
	
	
	__syncthreads();

}


__global__ void initilize_in(double *in,double *g,double *wdz,int ns1,int iz){

	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;

	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;

	long DIM;

	DIM=gridDim.x*gridDim.y*gridDim.z;
	
	in[j+i*DIM]=g[(j*ns1+iz-1)+i*DIM*ns1]*wdz[j+i*DIM];
	
	__syncthreads();

}

__global__ void initilize_in_go(double *in,double *g,double *wdz,int ns1,int iz){

	long i=blockIdx.y;

	long j=threadIdx.x+blockDim.x*blockIdx.x;

	long DIM=blockDim.x*gridDim.x;

	
	
	in[j+i*DIM]=g[(j*ns1+iz-1)+i*DIM*ns1]*wdz[j+i*DIM];
	
	__syncthreads();

}
__global__ void initilize_wdz(double *w,double *wdz,double ds2){
	
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long DIM,ij;
	
	DIM=gridDim.x*gridDim.y*gridDim.z;

	ij=j+i*DIM;
	
	wdz[ij]=exp(-w[ij]*ds2);
	
	__syncthreads();
}

__global__ void initilize_q(double *q,double *qInt,int ns1){
	
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long DIM;
	
	DIM=gridDim.x*gridDim.y*gridDim.z;
	
	q[j*ns1+DIM*ns1*i]=qInt[j+DIM*i];
	
	__syncthreads();
}

__global__ void initilize_q_inverse(double *q,double *qInt,int ns1){
	
	long i=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
	
	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long DIM;
	
	DIM=gridDim.x*gridDim.y*gridDim.z;
	
	q[j*ns1+DIM*ns1*i+ns1-1]=qInt[j+DIM*i];
	
	__syncthreads();
}


__global__ void sufaceField(cufftDoubleComplex *out,double *kxyzdz,int Nx){

	long ijk,ijkr;

	long offset=gridDim.x*gridDim.y*gridDim.z*threadIdx.x;

	ijk=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y+offset;

	ijkr=blockIdx.x+blockIdx.y*Nx+blockIdx.z*Nx*gridDim.y;

	out[ijk].x*=kxyzdz[ijkr];
	
	out[ijk].y*=kxyzdz[ijkr];

	__syncthreads();
}
__global__ void sufaceField_go(cufftDoubleComplex *out,double *kxyzdz,int Nxh1,int Nx,int Ny,int Nz){


	long ijk=threadIdx.x+blockDim.x*blockIdx.x;
	long NxNyNz=blockDim.x*gridDim.x;
	
	int iz=ijk/(Nxh1*Ny);
	int temp=ijk-iz*(Nxh1*Ny);
	int iy=temp/Nxh1;
	int ix=temp%Nxh1;

	long offset=NxNyNz*blockIdx.y;

	ijk=ix+iy*Nxh1+iz*Nxh1*Ny+offset;

	long ijkr=ix+iy*Nx+iz*Nx*Ny;

	out[ijk].x*=kxyzdz[ijkr];
	
	out[ijk].y*=kxyzdz[ijkr];

	__syncthreads();
}

__global__ void in_to_g(double *g,double *wdz_cu,cufftDoubleReal *in,int ns1,int iz){
	
	long i;
	
	long offset=gridDim.x*gridDim.y*gridDim.z*threadIdx.x;
	
	long NxNyNz=gridDim.x*gridDim.y*gridDim.z;
	
	i=blockIdx.x+blockIdx.y*gridDim.x+blockIdx.z*gridDim.x*gridDim.y;
	
	g[i*ns1+iz+offset*ns1]=in[(i)+offset]*wdz_cu[i+offset]/NxNyNz;

	__syncthreads();
}


__global__ void in_to_g_go(double *g,double *wdz_cu,cufftDoubleReal *in,int ns1,int iz){
	
	

	long NxNyNz=gridDim.x*blockDim.x;
	
	long offset=NxNyNz*blockIdx.y;
	
	long i=threadIdx.x+blockIdx.x*blockDim.x;

	
	
	g[i*ns1+iz+offset*ns1]=in[(i)+offset]*wdz_cu[i+offset]/NxNyNz;

	__syncthreads();
}



__global__ void minus_average(double *data,double *average_value){

	long i=threadIdx.x;

	long j=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;

	long DIM,ij;

	DIM=gridDim.x*gridDim.y*gridDim.z;
	
	ij=i*DIM+j;
	
	data[ij]=data[ij]-average_value[i]/(double)DIM;
	

}

__global__ void phi_w(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double hAB){
	
	
	long i=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long offset=threadIdx.x*gridDim.x*gridDim.y*gridDim.z;


	double eta;
	double psum;
	double waDiff,wbDiff;
	double wcmp,wopt;
	
	wopt=0.050;
	wcmp=0.10;
	
	eta=(wA_cu[i+offset]+wB_cu[i+offset]-hAB)/2;
	
	psum=1.0-phA_cu[i+offset]-phB_cu[i+offset];
		
	waDiff=hAB*phB_cu[i+offset]+eta-wA_cu[i+offset];

	wbDiff=hAB*phA_cu[i+offset]+eta-wB_cu[i+offset];

	waDiff-=wcmp*psum;

	wbDiff-=wcmp*psum;

	wA_cu[i+offset]+=wopt*waDiff;

	wB_cu[i+offset]+=wopt*wbDiff;
	
	
__syncthreads();
}

__global__ void phi_w_constrained(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB,double lambda){

	
	long i=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long offset=threadIdx.x*gridDim.x*gridDim.y*gridDim.z;
	
	
	double eta;
	double psum;
	double waDiff,wbDiff;
	double wcmp,wopt;
	
	wopt=0.050;
	wcmp=0.10;
	
	eta=(wA_cu[i+offset]+wB_cu[i+offset]-hAB)/2;
	
	psum=1.0-phA_cu[i+offset]-phB_cu[i+offset];
		
	waDiff=hAB*phB_cu[i+offset]+eta-wA_cu[i+offset]+lambda*(phA_cu[i+offset]-PhA_cu[i+offset]);

	wbDiff=hAB*phA_cu[i+offset]+eta-wB_cu[i+offset]+lambda*(phB_cu[i+offset]-(1-PhA_cu[i+offset]));
	
	waDiff-=wcmp*psum;
		
	wbDiff-=wcmp*psum;
	
	wA_cu[i+offset]+=wopt*waDiff;

	wB_cu[i+offset]+=wopt*wbDiff;
	
	
__syncthreads();
}

__global__ void phi_w_constrainedEx(double *wA_cu,double *wB_cu,double *phA_cu,double *phB_cu, double *PhA_cu,double hAB){

	long i=blockIdx.x+gridDim.x*blockIdx.y+gridDim.y*gridDim.x*blockIdx.z;
	
	long offset=threadIdx.x*gridDim.x*gridDim.y*gridDim.z;

	double eta,wum;
	double psum,psum1;
	double waDiff,wbDiff;
	double wcmp,wopt;
	
	wopt=1.60;
	wcmp=0.50;
	
	
	wum=-hAB*(PhA_cu[i+offset]*2-1)-(wA_cu[i+offset]-wB_cu[i+offset]);

	eta=(wA_cu[i+offset]+wB_cu[i+offset]-hAB+wum)/2;

	psum=1.0-phA_cu[i+offset]-phB_cu[i+offset];

	psum1=phA_cu[i+offset]-PhA_cu[i+offset];

	waDiff=hAB*(1-PhA_cu[i+offset])+eta-wA_cu[i+offset]-wum;

	wbDiff=hAB*PhA_cu[i+offset]+eta-wB_cu[i+offset];

	waDiff-=2.0*wcmp*psum-2.0*wcmp*psum1;//+wcmp*fpsum1*5;

	wbDiff-=2*wcmp*psum1;

	wA_cu[i+offset]+=wopt*waDiff;

	wB_cu[i+offset]+=wopt*wbDiff;

__syncthreads();
}






