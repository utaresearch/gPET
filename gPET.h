#ifndef __GPET_H__
#define __GPET_H__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include "cutil_math.h"
#include <cuda.h>
#include <cublas.h>
#include <curand_kernel.h>


#define FILEEXIST(x) do{if(!x){printf("FILE EXIST ERROR at %s:%d\n",__FILE__,__LINE__);exit(EXIT_FAILURE);}}while(0)
#define CUDA_CALL(x) do{if(x != cudaSuccess){printf("CUDA Error at %s:%d\n",  __FILE__,__LINE__);exit(EXIT_FAILURE);}}while(0)

typedef struct//phantom
{   
   int   *mat;
   float *dens;
   int   Unxvox, Unyvox, Unzvox;
   float dx, dy, dz;
   float Offsetx, Offsety, Offsetz;
   float Sizex, Sizey, Sizez;

}Phantom;

typedef struct
{   
   float3 *xbuffer;
   float4 *vxbuffer;  // xbuffer: x,y,z,w;vxbuffer: vx,vy,vz,E;
   int* eventid;
   double *time;
  int NParticle;
}Particle;

typedef struct
{
   int Ntype;
   float *halftime, *decayRatio, *coef;
}Isotopes;

typedef struct
{
   int NSource;
   int *type, *shape;
   unsigned int *natom;
   float *shapecoeff;
}Source;

typedef struct object_t
{ 
// panel index
   int panel;
// panel dimension
   float lengthx, lengthy, lengthz; 
// module size
   float MODx, MODy, MODz;
// module space size
   float Mspacex, Mspacey, Mspacez;
// LSO size
   float LSOx, LSOy, LSOz;
// space size
   float spacex, spacey, spacez;
// offset (top surface center, local coordinate origin) of each module
   float offsetx, offsety, offsetz;
// module local direction
   float directionx, directiony, directionz;
// unit vector along x direction of each module
   float UniXx, UniXy, UniXz;
// unit vector along y direction of each module
   float UniYx, UniYy, UniYz;
// unit vector along z direction of each module
   float UniZx, UniZy, UniZz;
} OBJECT; // name of the structure

typedef struct object_v
{ // material index
   int material;
// material density
   float density;
} OBJECT_V;

typedef struct Event
{
    int parn,pann,modn,cryn,siten,eventid;//siten: the index in current depth
    double t;
    float E, x,y,z;
} Event;

struct compare_parn
{
    __host__ __device__ bool operator()(Event a, Event b)
    {
        return a.parn < b.parn;
    }
};
struct compare_siten
{
    __host__ __device__ bool operator()(Event a, Event b)
    {
        return a.siten < b.siten;
    }
};
struct compare_t
{
    __host__ __device__ bool operator()(Event a, Event b)
    {
        return a.t < b.t;
    }
};


/***************************************************
the following are declaration of function
***************************************************/
inline __host__ __device__ int ind2To1(int i, int j, int nx, int ny)
//      convert a voxel real indices into a single index
{
//      different ordering
        return i*ny+j;
//      return j*nx+i;
}

//in iniDevice.cu
void iniDevice(int deviceNo);
void printDevProp(int device);

//in initilize.cu
Isotopes loadIsotopes();
Phantom loadPhantom(char matfile[100], char denfile[100], int* pdim, float* poffset, float* psize);
Particle readParticle(char sourcefile[100],int NParticle);
Source readSource(char sourcefile[100]);
void spline(float *X, float *Y, float *A, float *B, float *C, float *D, float S1, float SN, int N);
void inirngG();
void rmater(float *eminph, float *emax);
void rlamph();
void rcompt();
void rcmpsf();
void rphote();
void rrayle();
void rrayff();
float itphip(int matid, float e);
void initPhantom(Phantom phantom);
void init(Phantom phantom);
void iniwck(float eminph,float emax, Phantom phantom);
void iniwck(float eminph,float emax, struct object_v* objectMaterial);
void iniPanel(struct object_t* objectArray, struct object_v* objectMaterial,int totalOb);

//in gPET.cu
void simulateParticle(Particle particle,int ptype_h, int total_Panels);
void sampleParticle(Source source, Isotopes isotopes, float tstart, float tend, int total_Panels);
float findT(float tstart, float tend, Isotopes isotopes, Source source);

//in detect.cu
struct object_t InitializeObject();
void read_file_ro(struct object_t** objectArray, struct object_v** objectMaterial, int* total_Panels, char fname[100]);
void quicksort(Event*  hits,int start, int stop, int sorttype);
void orderevents(int* counts,Event* events_d);
void outputData(void *src, const int size, const char *outputfilename, const char *mode);
void quicksort_d(Event* events_d,int start, int stop, int sorttype);
void quicksort_h(Event* events_d,int start, int stop, int sorttype);
int outevents(int* num_d, Event* totalevents_d, const char *outputfilename);

//in gPET_kernals.cu
__global__ void setupcuseed(int* seed);
__device__ int4 getAbsVox(float3 xtemp);
__device__ float lamwck(float e);
__device__ float lamwckde(float e);
__device__ float itphip_G(int matid, float e);
__device__ float irylip(int matid, float e);
__device__ float icptip(int matid, float e);
__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe, int matid);
__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe);
__device__ void rylsam(float energytemp, int matid, curandState *localState_pt, float *costhe);
__device__ float getDistance(float3 coords, float4 direcs);
__device__ void rotate(float *u, float *v, float *w, float costh, float phi);
__global__ void photon(const int nactive);
__device__ float3 setPositronRange(float3 xtemp, float4 vxtemp, curandState* plocalState,int usedirection);
__device__ float sampleEkPositron(int type, float* d_coef, curandState* plocalState);
__device__ float3 getPositionFromShape(int sourceindex, int shapeindex, float* shapecoeff, curandState* plocalState);
__global__ void setPosition(int nsource, unsigned int totalatom, float tref, float t, unsigned int* d_natom, unsigned int* d_sumpartial, int* d_type, int* d_shape, 
    float* d_shapecoeff, float* d_halftime, float* d_decayRatio, float* d_coef, int useprange);
__global__ void setPositionForPhoton(int total,int curpar, int useprange);
__global__ void setSitenum(int total, Event* events_d,int depth);
__global__ void energywindow(int* counts, Event* events,int total, float thresholder, float upholder);
__global__ void deadtime(int* counts,Event* events,int total, float interval, int deadtype);
__global__ void addnoise(int* counts, Event* events_d, float lambda, float Emean, float sigma, float interval);
__device__ int adder(int* counts_d, Event* events_d, Event event);
__device__ int readout(int* counts_d, Event* events_d,int depth, int policy);
__global__ void blur(int total, Event* events, int Eblurpolicy, float Eref, float Rref, float slope, float Spaceblur);
__global__ void photonde(Event* events_d, int* counts_d, int nactive, int bufferID, float* dens, int *mat, int *panelID, float *lenx, float *leny, float *lenz,
                       float *MODx, float *MODy, float *MODz, float *Msx, float *Msy, float *Msz, float *LSOx, float *LSOy, float *LSOz, float *sx, float *sy, float *sz,
                       float *ox, float *oy, float *oz, float *dx, float *dy, float *dz, float *UXx, float *UXy, float *UXz,
                       float *UYx, float *UYy, float *UYz,float *UZx, float *UZy, float *UZz);
__device__ void crystalSearch(float3 xtemp2,float leny_S,float lenz_S,float MODy_S,float MODz_S,float Msy_S,float Msz_S,float LSOy_S,float LSOz_S,float sy_S,float sz_S,float dy_S, float dz_S, int *m_id, int *M_id, int *L_id);
//*/
#endif


