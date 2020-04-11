#ifndef __GPET_KERNALSH__
#define __GPET_KERNALSH_

#include "constants.h"
__device__ curandState cuseed[NRAND];
__device__ float4 vx_gBrachy[NPART];
__device__ float3 x_gBrachy[NPART];
__device__ double d_time[NPART];
__device__ int d_eventid[NPART];
__device__ unsigned int d_curemitted;
__constant__ int moduleN,crystalN,moduleNy,crystalNy;
__constant__ float surface_d[10*MAXSURFACE];
__constant__ float recordsphere_d[4];
__constant__ float dx_gBrachy,dy_gBrachy,dz_gBrachy;
__constant__ int Unxvox,Unyvox,Unzvox;
__constant__ float idx_gBrachy,idy_gBrachy,idz_gBrachy;
__constant__ float Offsetx_gBrachy,Offsety_gBrachy,Offsetz_gBrachy;
__constant__ int rdepth_d, rpolicy_d;

__device__ int nsstk;
__device__ float sf[NSSTACK], NonAngle_d;
__device__ int sid[NSSTACK];

__constant__ float eabsph;
__constant__ int Nsurface_d;

cudaArray *mat;
cudaArray *dens;
texture<int,3,cudaReadModeElementType> mat_tex;
texture<float,3,cudaReadModeElementType> dens_tex;

//read file with prefix pre4phot for photon
__constant__ int nmat;


__device__ __constant__ float idleph;
__device__ __constant__ float elaph0;

texture<float,1,cudaReadModeElementType> lamph_tex;

__device__ __constant__ float idlecp;
__device__ __constant__ float ecmpt0;

texture<float,1,cudaReadModeElementType> compt_tex;


__device__ __constant__ float idcpcm,idecm;

texture<float, 3, cudaReadModeElementType> s_tex;

__device__ __constant__ float idlepe;
__device__ __constant__ float ephte0;

texture<float,1,cudaReadModeElementType> phote_tex;

__device__ __constant__ float idlerl;
__device__ __constant__ float erayl0;

texture<float,1,cudaReadModeElementType> rayle_tex;

__device__ __constant__ float idcprl,iderl;

texture<float, 3, cudaReadModeElementType> f_tex;

//woodcock
__constant__ float idlewk;
__constant__ float wcke0;

texture<float,1,cudaReadModeElementType> woock_tex;

texture<float,1,cudaReadModeElementType> woockde_tex;

__constant__ int dev_totalPanels;

#endif