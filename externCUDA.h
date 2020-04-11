#include "constants.h"

extern __device__ curandState cuseed[NRAND];
extern __device__ float4 vx_gBrachy[NPART];
extern __device__ float3 x_gBrachy[NPART];
extern __device__ double d_time[NPART];
extern __device__ int d_eventid[NPART];
extern __device__ unsigned int d_curemitted;
extern __constant__ int Nsurface_d, rdepth_d, rpolicy_d;
//==========================================================
//      GPU configurations
//==========================================================
extern float Eref, Rref, Eslope, Sblur, Eth, Ewinmin, Ewinmax, dtime;
extern int blurpolicy, dlevel, dtype, useprange_h;
extern float emax, eminph;

extern __constant__ int moduleN,crystalN,moduleNy,crystalNy;
extern __constant__ float surface_d[10*MAXSURFACE];
extern __constant__ float recordsphere_d[4], NonAngle_d;

//for phantom
extern __constant__ float dx_gBrachy,dy_gBrachy,dz_gBrachy;
extern __constant__ int Unxvox,Unyvox,Unzvox;
extern __constant__ float idx_gBrachy,idy_gBrachy,idz_gBrachy;
extern __constant__ float Offsetx_gBrachy,Offsety_gBrachy,Offsetz_gBrachy;


// common variable group for panel geometry
extern __constant__ int dev_totalPanels;
extern 	float *dens_panel;
extern 	int *mat_panel;
extern 	int *panelID;
extern 	float *lengthx_panel, *lengthy_panel, *lengthz_panel;
extern 	float *MODx_panel, *MODy_panel, *MODz_panel;
extern 	float *Mspacex_panel, *Mspacey_panel, *Mspacez_panel;
extern 	float *LSOx_panel, *LSOy_panel, *LSOz_panel;
extern 	float *spacex_panel, *spacey_panel, *spacez_panel;
extern 	float *offsetx_panel, *offsety_panel, *offsetz_panel;
extern 	float *directionx_panel, *directiony_panel, *directionz_panel;
extern 	float *UniXx_panel, *UniXy_panel, *UniXz_panel;
extern 	float *UniYx_panel, *UniYy_panel, *UniYz_panel;
extern 	float *UniZx_panel, *UniZy_panel, *UniZz_panel;

extern 	__device__ int nsstk;
extern 	__device__ float sf[NSSTACK];
extern 	__device__ int sid[NSSTACK];

extern float eabsph_h;
extern __constant__ float eabsph; 

extern __device__ __constant__ int nmat;

extern cudaArray *mat;
extern cudaArray *dens;
extern texture<int,3,cudaReadModeElementType> mat_tex;
extern texture<float,3,cudaReadModeElementType> dens_tex;

extern float elaph_h[NLAPH],lamph_h[NLAPH*MAXMAT],lampha_h[NLAPH*MAXMAT],lamphb_h[NLAPH*MAXMAT],
	lamphc_h[NLAPH*MAXMAT],lamphd_h[NLAPH*MAXMAT];


extern __device__ __constant__ float idleph;
extern __device__ __constant__ float elaph0;
extern float idleph_h;
extern cudaArray *lamph;
extern texture<float,1,cudaReadModeElementType> lamph_tex;

extern __device__ __constant__ float idlecp;
extern __device__ __constant__ float ecmpt0;
extern float idlecp_h;
extern float ecmpt_h[NCMPT],compt_h[NCMPT*MAXMAT];
extern cudaArray *compt;
extern texture<float,1,cudaReadModeElementType> compt_tex;

extern float idcpcm_h, idecm_h;
extern __device__ __constant__ float idcpcm,idecm;
extern float mucmpt_h[NCPCM*NECM*MAXMAT];
extern cudaArray* sArray;
extern texture<float, 3, cudaReadModeElementType> s_tex;

extern __device__ __constant__ float idlepe;
extern __device__ __constant__ float ephte0;
extern float idlepe_h;
extern float ephte_h[NPHTE],phote_h[NPHTE*MAXMAT];
extern cudaArray *phote;
extern texture<float,1,cudaReadModeElementType> phote_tex;

extern __device__ __constant__ float idlerl;
extern __device__ __constant__ float erayl0;
extern float idlerl_h;
extern float erayl_h[NRAYL],rayle_h[NRAYL*MAXMAT];
extern cudaArray *rayle;	//	cross section data
extern texture<float,1,cudaReadModeElementType> rayle_tex;

extern float idcprl_h, iderl_h;
extern __device__ __constant__ float idcprl,iderl;
extern float murayl_h[NCPRL*NERL*MAXMAT];
extern cudaArray* fArray;
extern texture<float, 3, cudaReadModeElementType> f_tex;

//woodcock
extern __device__ __constant__ float idlewk;
extern __device__ __constant__ float wcke0;
extern float idlewk_h, wcke0_h;
extern float woock_h[NWCK];
extern cudaArray *woock;
extern texture<float,1,cudaReadModeElementType> woock_tex;

extern cudaArray *woockde;
extern texture<float,1,cudaReadModeElementType> woockde_tex;
