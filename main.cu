// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
using namespace std;

// includes, project
#include "cutil_math.h"
#include <cuda.h>
#include <cublas.h>
#include <curand_kernel.h>

// includes
//#include "cuPrintf.cu"
#include "gPET.h"
#include "gPETInternal.h"
#include "externCUDA.h"

/****************************************************
        main program
****************************************************/
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Please execute ./gPET input_file\n");
        printf("Thanks.\n\n");
        exit(1);
    }

    printf("***************************************\n");
    printf("Computation starts\n");
    printf("***************************************\n");

    clock_t start_time, end_time;
    float time_diff;    
    start_time = clock();

    char denfile[100], matfile[100],sourcefile[100],detectgeo[100], buffer[200];;
    int usepsf=0, NParticle=0, ptype=-1, deviceNo=0, Nsurface=1;
    float tstart=0, tend=1, nonAngle = 0;//s
    int pdim[3];
    float poffset[3], psize[3], recordsphere[4];

//reading configuration
    FILE *configFile = fopen(argv[1], "r");

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d\n", &deviceNo);
    printf("%d\n", deviceNo);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%f\n", &nonAngle);
    printf("%f\n", nonAngle);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    for(int i=0;i<3;i++)
    {
        fscanf(configFile, "%d ",&pdim[i]);
        printf("%d ",pdim[i]);
    }

    fgets(buffer, 200, configFile);
    printf("\n%s\n", buffer);
    for(int i=0;i<3;i++)
    {
        fscanf(configFile, "%f ",&poffset[i]);
        printf("%f ",poffset[i]);
    }

    fgets(buffer, 200, configFile);
    printf("\n%s\n", buffer);
    for(int i=0;i<3;i++)
    {
        fscanf(configFile, "%f ",&psize[i]);
        printf("%f ",psize[i]);
    }

    fgets(buffer, 200, configFile);
    printf("\n%s\n", buffer);
    fscanf(configFile, "%s\n", matfile);
    printf("%s\n", matfile);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%s\n", denfile);
    printf("%s\n", denfile);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d\n", &NParticle);
    printf("%d\n", NParticle);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d\n", &usepsf);
    printf("%d\n", usepsf);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%s\n", sourcefile);
    printf("%s\n", sourcefile);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d\n", &ptype);
    printf("%d\n", ptype);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d\n", &useprange_h);
    printf("%d\n", useprange_h);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%f %f\n", &tstart, &tend);
    printf("%f %f\n", tstart, tend);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    for(int i=0;i<4;i++)
    {
        fscanf(configFile, "%f ",&recordsphere[i]);
        printf("%f ",recordsphere[i]);
    }

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%f\n", &eabsph_h);
    printf("%f\n", eabsph_h);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%s\n", detectgeo);
    printf("%s\n", detectgeo);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d\n", &Nsurface);
    float surface[10*Nsurface];
    for(int tmp = 0; tmp<Nsurface; tmp++)
    {
        for(int i=0;i<10;i++)
        {
            fscanf(configFile, "%f ",&surface[tmp*10+i]);
            printf("%f ",surface[tmp*10+i]);
        }
        fscanf(configFile, "\n");
    }
        
    int rdepth, rpolicy;
    fgets(buffer, 200, configFile);
    printf("\n%s\n", buffer);
    fscanf(configFile, "%d %d\n", &rdepth, &rpolicy);
    printf("%d %d\n", rdepth, rpolicy);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%f\n", &Eth);
    printf("%f\n", Eth);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d %f %f %f %f\n", &blurpolicy, &Eref, &Rref, &Eslope, &Sblur);
    printf("%d %f %f %f %f\n", blurpolicy, Eref, Rref, Eslope, Sblur);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%d %d %f\n", &dlevel, &dtype, &dtime);
    printf("%d %d %f\n", dlevel, dtype, dtime);

    fgets(buffer, 200, configFile);
    printf("%s\n", buffer);
    fscanf(configFile, "%f %f\n", &Ewinmin, &Ewinmax);
    printf("%f %f\n", Ewinmin, Ewinmax);

    fclose(configFile);

    if(Nsurface>MAXSURFACE) 
    {
        printf("Please increase MAXSURFACE in constants.h\n");
        return 1;
    }
//initialize device with index deviceNo    
    iniDevice(deviceNo);

//copying data from CPU to GPU, CUDA_CALL is defined in gPET.h
    CUDA_CALL(cudaMemcpyToSymbol(NonAngle_d, &nonAngle, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(rpolicy_d, &rpolicy, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(rdepth_d, &rdepth, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(Nsurface_d, &Nsurface, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(surface_d, surface, sizeof(float)*10*Nsurface));
    CUDA_CALL(cudaMemcpyToSymbol(recordsphere_d, recordsphere, sizeof(float)*4));

//get phantom information in CPU      
    Phantom phantom = loadPhantom(matfile,denfile,pdim,poffset,psize); 

//initialize GPU memory, setting up the random number seed, phantom information, and necessary texture information
    init(phantom);
    struct object_t* panelArray;
    struct object_v* panelMaterial;
    int total_Panels=0;
    read_file_ro(&panelArray,&panelMaterial,&total_Panels,detectgeo);
    iniPanel(panelArray,panelMaterial,total_Panels);
    iniwck(eminph,emax,panelMaterial); 
    end_time = clock(); 
    time_diff = ((float)end_time - (float)start_time)/CLOCKS_PER_SEC;   
    printf("\n\n****************************************\n");
    printf("Initialize time: %f s.\n\n",time_diff);

//get particle information from either PSF or sampling from source
    if(usepsf) 
    {
        Particle particle = readParticle(sourcefile,NParticle);
        simulateParticle(particle,ptype,total_Panels);
    }
    else
    {
        Isotopes isotopes= loadIsotopes();
        Source source = readSource(sourcefile);
        sampleParticle(source, isotopes, tstart, tend, total_Panels);
    }  

//free memory
    cudaUnbindTexture(mat_tex) ;
    cudaFreeArray(mat) ;

    cudaUnbindTexture(dens_tex) ;
    cudaFreeArray(dens) ;

    cudaUnbindTexture(lamph_tex) ;
    cudaFreeArray(lamph) ;

    cudaUnbindTexture(compt_tex);
    cudaFreeArray(compt);
    cudaUnbindTexture(s_tex) ;
    cudaFreeArray(sArray) ;

    cudaUnbindTexture(phote_tex);
    cudaFreeArray(phote);

    cudaUnbindTexture(rayle_tex) ;
    cudaFreeArray(rayle) ;
    cudaUnbindTexture(f_tex);
    cudaFreeArray(fArray);

    cudaUnbindTexture(woock_tex);
    cudaFreeArray(woock);
    cudaUnbindTexture(woockde_tex);
    cudaFreeArray(woockde);

    cudaFree(cuseed);
    cudaFree(sf);
    cudaFree(sid);
    cudaFree(x_gBrachy);
    cudaFree(vx_gBrachy);
    cudaFree(d_eventid);
    cudaFree(d_time);
/***************************************************/
    end_time = clock(); 
    time_diff = ((float)end_time - (float)start_time)/CLOCKS_PER_SEC;   
    printf("\n\n****************************************\n");
    printf("Total time: %f s.\n\n",time_diff);   
    printf("****************************************\n\n\n");               
    printf("Have a nice day!\n");

    cudaDeviceReset();
    return 0;  
}   
