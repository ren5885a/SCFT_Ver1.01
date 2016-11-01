#include"init.h"
#include<stdio.h>
/*
	 COMMAND LINE ARGUMENTS
	
*/

int init_scft(CUFFT_INFO *cufft_info,GPU_INFO *gpu_info,int argc, char **argv){
	
	char *typeChoice = 0;
	
	gpu_info->kernal_type=-1;
	
	getCmdLineArgumentString(argc, (const char **) argv, "kernal", &typeChoice);
	
	cufft_info->MaxIT=getCmdLineArgumentInt(argc, (const char **) argv, "step");

	
	
	gpu_info->GPU_N=getCmdLineArgumentInt(argc, (const char **) argv, "GPU_N");
	char comment[200];
	if(gpu_info->GPU_N==0) gpu_info->GPU_N=1;
	//printf("Maximal iteritive step is %d\n",cufft_info->MaxIT);

	gpu_info->whichGPUs=(int*)malloc(sizeof(int)*(gpu_info->GPU_N));

	for(int i=0;i<(gpu_info->GPU_N);i++){

		char gpu_id[20];
		
		sprintf(gpu_id,"gpu%d",i);
		
		gpu_info->whichGPUs[i]=i;	//!Define on these GPU to calculate 
		
		gpu_info->whichGPUs[i]=getCmdLineArgumentInt(argc, (const char **) argv, gpu_id);

	}

	cufft_info->AverIt=10;	

	if(cufft_info->MaxIT<1){
		
		printf("Maximal iterative step did not set, defalt set as 100.\n");
		
		cufft_info->MaxIT=100;
		
		
			
		printf("It could as well be set by adding flag -step=Num. \n");

	}
	if (0 != typeChoice){
		
        	if (!strcasecmp(typeChoice, "GPU")){
		
			gpu_info->kernal_type = 1;
		
        	}
        	else if (!strcasecmp(typeChoice, "CPU")){
			
        		gpu_info->kernal_type = 2;
			
        	}
       		else{
		
            		gpu_info->kernal_type = -1;
			
        	}
    	}
	else{

			gpu_info->kernal_type = 1;
	}
	
	if(gpu_info->kernal_type==1){
	
		printf("In this programm, we use GPU.\n");
		
		FILE *fp;
		
		if((fp=fopen("ab.txt","r"))==false){
			printf("Configration file ab.txt did not exist, please check it in your directory.\n");
		}

	
		fscanf(fp,"intag=%d\n",&cufft_info->intag);		//in=1: inputing configuration is given;
		fscanf(fp,"xN=%lf\n",&cufft_info->hAB);
		fscanf(fp,"fa=%lf\n", &cufft_info->fA);
		fscanf(fp,"lx=%lf, ly=%lf, lz=%lf\n",&cufft_info->lx, &cufft_info->ly, &cufft_info->lz);
		fscanf(fp,"Nx=%d, Ny=%d, Nz=%d\n",&cufft_info->Nx, &cufft_info->Ny, &cufft_info->Nz);
		fscanf(fp,"ds0=%lf\n", &cufft_info->ds0);
		fscanf(fp,"batch=%d\n", &cufft_info->Num_SCFT);

		cufft_info->batch=cufft_info->Num_SCFT;
		
		//! output configuration.
		/*
		printf("intag=%d\n",cufft_info->intag);
		printf("xN=%g\n",cufft_info->hAB);
		printf("%d %d %d\n",cufft_info->Nx,cufft_info->Ny,cufft_info->Nz);	
		printf("lx=%lf, ly=%lf, lz=%lf\n",cufft_info->lx, cufft_info->ly, cufft_info->lz);
		printf("%lf\n", cufft_info->ds0);
		printf("batch=%d\n",cufft_info->Num_SCFT);
		
		
		printf("Maximum iterative step is %d \n",cufft_info->MaxIT);
		*/
		fclose(fp);

		init_cuda(gpu_info,0);
		
		initialize_cufft(gpu_info,cufft_info);

		

	
	}
		
	else if(gpu_info->kernal_type==2)
	
		printf("In this programm, we use CPU.\n");
	
	else if(gpu_info->kernal_type==-1){

		printf("Set kernal type by adding flag -kernal=GPU/CPU, but CPU now is not aviliable\n");
		
	}
	
	
	
	
}
