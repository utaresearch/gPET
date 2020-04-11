#ifndef __CONSTANTS__
#define __CONSTANTS__

#define RECORDPSF 1
#define OUTPUTPSF 1
#define OUTPUTHIT 1
#define RECORDRayleigh 0
#define BISEARCH 0
#define MAXMAT 10 //max number of materials
#define NWCK 4096 // number of energy entries for woock method

//The following cannot be changed
#define NPART  524288 // simulate particle number in a batch
#define NTHREAD_PER_BLOCK_GPET 256 //cannot be too large because of shared memory
#define NSSTACK 15*NPART
#define NSSTACKSHARED (15*NTHREAD_PER_BLOCK_GPET)
#define NRAND  262144 //random number array, should be divisible by 512 since 512 is used in gPET.cu
#define MAXT 1e20
#define MAXSURFACE 5 //max number of surfaces to exclude hit events
#define NLAPH 4096  //number of energy entries in cross section data, see corresponding file
#define NCMPT 4096 //compton
#define NCPCM 301
#define NECM 151
#define NPHTE 4096  //photoelectric
#define NRAYL 4096  //rayleigh
#define NCPRL 301
#define NERL 151
//==========================================================
//      physical and mathematical constants
//==========================================================
#define PI 3.1415926535897932384626433f
#define TWOPI 6.2831853071795864769252867f

#define MC2 510.9991e3
#define IMC2 1.95695060911e-6
#define ZERO 1.0e-20
#define SZERO 1.0e-4

#define INF 1.0e20

#endif