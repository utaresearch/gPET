#ifndef __INITIALIZE_CU__
#define __INITIALIZE_CU__
#include "gPET.h"
#include "externCUDA.h"

int nmat_h;

using namespace std;

Isotopes loadIsotopes()
{
    Isotopes isotopes;
    ifstream infile("data/isotopes.txt");
    FILEEXIST(infile);
    printf("reading /data/isotopes.txt\n");
    infile >> isotopes.Ntype;
    infile.ignore(512,'#');
    isotopes.halftime=new float[isotopes.Ntype];
    isotopes.decayRatio=new float[isotopes.Ntype];
    isotopes.coef=new float[isotopes.Ntype*8];
    for(int i=0;i<isotopes.Ntype;i++)
    {
        infile >> isotopes.halftime[i]>>isotopes.decayRatio[i]>>isotopes.coef[8*i]>>isotopes.coef[8*i+1]>>isotopes.coef[8*i+2];
        infile >> isotopes.coef[8*i+3]>>isotopes.coef[8*i+4]>>isotopes.coef[8*i+5]>>isotopes.coef[8*i+6]>>isotopes.coef[8*i+7];
        cout << i <<" "<< isotopes.halftime[i]<<" " <<isotopes.decayRatio[i]<<" "<< \
        isotopes.coef[8*i]<<" "<< isotopes.coef[8*i+7]<<endl;
    }
    printf("finish read isotopes;\n\n");        
    infile.close();       
    return isotopes;
}
Phantom loadPhantom(char matfile[100], char denfile[100],int* pdim, float* poffset, float* psize)
/*******************************************************************
c*    Reads voxel geometry from an input file                      *               *
c******************************************************************/
{
    Phantom phantom;
    cout<<"\n\nloading phantom\n";
    
    phantom.Unxvox = pdim[0];
    phantom.Unyvox = pdim[1];
    phantom.Unzvox = pdim[2];  
    printf("CT dimension: %f %f %f\n", phantom.Unxvox, phantom.Unyvox, phantom.Unzvox);
    
    phantom.Offsetx = poffset[0];
    phantom.Offsety = poffset[1];
    phantom.Offsetz = poffset[2];
    printf("CT offset: %f %f %f\n", phantom.Offsetx, phantom.Offsety, phantom.Offsetz);
    
    phantom.Sizex = psize[0];
    phantom.Sizey = psize[1];
    phantom.Sizez = psize[2];   
    printf("CT Size: %f %f %f\n", phantom.Sizex, phantom.Sizey, phantom.Sizez);

    int numvox=phantom.Unxvox*phantom.Unyvox*phantom.Unzvox;
    phantom.mat = new int[numvox];
    ifstream infilemat(matfile,ios::binary);
    FILEEXIST(infilemat);
    infilemat.read(reinterpret_cast <char*> (&phantom.mat[0]), sizeof(int)*numvox);
    infilemat.close();

    phantom.dens = new float[numvox];
    ifstream infileden(denfile,ios::binary);
    FILEEXIST(infileden);
    infileden.read(reinterpret_cast <char*> (&phantom.dens[0]), sizeof(float)*numvox);
    infileden.close();

    phantom.dx = phantom.Sizex/phantom.Unxvox; 
    phantom.dy = phantom.Sizey/phantom.Unyvox; 
    phantom.dz = phantom.Sizez/phantom.Unzvox; 
    cout<<"finish loading phantom"<<endl;
    cout<<"resolution is "<<phantom.dx << phantom.dy<< phantom.dz<<endl;
    return phantom;
}

Particle readParticle(char sourcefile[100],int NParticle)
{
    Particle particle;
    double data[8];

    ifstream infile(sourcefile,ios::binary);
    FILEEXIST(infile);
    printf("reading %s\n", sourcefile);
    int start, stop;
    start=infile.tellg();
    infile.seekg(0, ios::end);
    stop=infile.tellg();
    if(NParticle>(stop-start)/64)
    {
        cout<<"Do not have enough particles in PSF, Changing simulated number from "<<NParticle<<" to "<< (stop-start)/64 <<endl;
        NParticle = (stop-start)/64;
    }
    particle.xbuffer=new float3[NParticle];
    particle.vxbuffer=new float4[NParticle];
    particle.eventid=new int[NParticle];
    particle.time=new double[NParticle];
    infile.seekg(0, ios::beg);
    for(int i=0;i<NParticle;i++)
    {
        infile.read(reinterpret_cast <char*> (&data), sizeof(data));
        particle.xbuffer[i]=make_float3(data[0],data[1],data[2]);
        particle.vxbuffer[i]=make_float4(data[4],data[5],data[6],data[7]);
        particle.eventid[i]=i;
        particle.time[i] = data[3];        
        if(i<6)
        {
            printf("the first %d particle: %f %f %f\n",i,particle.xbuffer[i].x,particle.vxbuffer[i].x,particle.time[i] );
        }
    }
    printf("finish read: source PSF;\n\n");        
    infile.close();

    particle.NParticle = NParticle;       
    return particle;
}
Source readSource(char sourcefile[100])
{
    Source source;

    ifstream infile(sourcefile);
    FILEEXIST(infile);
    printf("reading %s\n", sourcefile);
    infile >> source.NSource;
    cout<< source.NSource<<"\n";
    infile.ignore(512,'#');
    source.natom=new unsigned int[source.NSource];
    source.type=new int[source.NSource];
    source.shape=new int[source.NSource];
    source.shapecoeff=new float[source.NSource*6];
    for(int i=0;i<source.NSource;i++)
    {
        infile >> source.natom[i] >> source.type[i] >> source.shape[i];
        cout<< i <<" "<< source.natom[i]<<" " << source.type[i]<<" "<< source.shape[i];
        for(int j=0;j<6;j++)
        {
            infile>>source.shapecoeff[6*i+j];
            cout<<" "<<source.shapecoeff[6*i+j];
        }
        cout<<endl;        
    }//*/
    printf("finish read: source;\n\n");        
    infile.close();       
    return source;
}

void spline(float *X, float *Y, float *A, float *B, float *C, float *D, float S1, float SN, int N)
//  possible error from FORTRAN to C

/*  CUBIC SPLINE INTERPOLATION BETWEEN TABULATED DATA.

C  INPUT:
C     X(I) (I=1, ...,N) ........ GRID POINTS.
C                     (THE X VALUES MUST BE IN INCREASING ORDER).
C     Y(I) (I=1, ...,N) ........ CORRESPONDING FUNCTION VALUES.
C     S1,SN ..... SECOND DERIVATIVES AT X(1) AND X(N).
C             (THE NATURAL SPLINE CORRESPONDS TO TAKING S1=SN=0).
C     N ........................ NUMBER OF GRID POINTS.
C
C     THE INTERPOLATING POLYNOMIAL IN THE I-TH INTERVAL, FROM
C  X(I) TO X(I+1), IS PI(X)=A(I)+X*(B(I)+X*(C(I)+X*D(I))).
C
C  OUTPUT:
C     A(I),B(I),C(I),D(I) ...... SPLINE COEFFICIENTS.
C
C     REF.: M.J. MARON, 'NUMERICAL ANALYSIS: A PRACTICAL
C           APPROACH', MACMILLAN PUBL. CO., NEW YORK 1982.
C*************************************************************/
{
//  linear interpolation, you can use the for loop here and comment the following lines.
    /*  for(int i = 0; i< N-1; i++)
        {
            B[i] = (Y[i+1]-Y[i])/(X[i+1]-X[i]);
            A[i] = (Y[i]*X[i+1] - X[i]*Y[i+1])/(X[i+1]-X[i]);
            C[i] = 0.0;
            D[i] = 0.0;
        }*/

    //IMPLICIT DOUBLE PRECISION (A-H,O-Z)
    //  DIMENSION X(N),Y(N),A(N),B(N),C(N),D(N)

    if(N < 4)
    {
        printf("SPLINE INTERPOLATION CANNOT BE PERFORMED WITH %d POINTS. STOP.\n",N);
        exit(1);
    }

    int N1 = N-1;
    int N2 = N-2;
//  AUXILIARY ARRAYS H(=A) AND DELTA(=D).
    for(int i = 0; i < N1; i++)
    {
        if(X[i+1]-X[i] < 1.0e-10)
        {
            printf("SPLINE X VALUES NOT IN INCREASING ORDER. STOP.\n");
            exit(1);
        }
        A[i] = X[i+1] - X[i];
        D[i] = (Y[i+1] - Y[i])/A[i];
    }

//  SYMMETRIC COEFFICIENT MATRIX (AUGMENTED).
    for(int i = 0; i < N2; i++)
    {
        B[i] = 2.0F * (A[i] + A[i+1]);
        int k = N1 - i - 1;
        D[k] = 6.0F * (D[k] - D[k-1]);
    }

    D[1] -= A[0] * S1;
    D[N1-1] -= A[N1-1] * SN;
//  GAUSS SOLUTION OF THE TRIDIAGONAL SYSTEM.
    for(int i = 1; i < N2; i++)
    {
        float R = A[i]/B[i-1];
        B[i] -= R * A[i];
        D[i+1] -= R * D[i];
    }
//  THE SIGMA COEFFICIENTS ARE STORED IN ARRAY D.
    D[N1-1] = D[N1-1]/B[N2-1];
    for(int i = 1; i < N2; i++)
    {
        int k = N1 - i - 1;
        D[k] = (D[k] - A[k] * D[k+1])/B[k-1];
    }
    D[N-1] = SN;
//  SPLINE COEFFICIENTS.
    float SI1 = S1;
    for(int i = 0; i < N1; i++)
    {
        float SI = SI1;
        SI1 = D[i+1];
        float H = A[i];
        float HI = 1.0F/H;
        A[i] = (HI/6.0F)*(SI*X[i+1]*X[i+1]*X[i+1]-SI1*X[i]*X[i]*X[i])
               +HI*(Y[i]*X[i+1]-Y[i+1]*X[i])
               +(H/6.0F)*(SI1*X[i]-SI*X[i+1]);
        B[i] = (HI/2.0F)*(SI1*X[i]*X[i]-SI*X[i+1]*X[i+1])
               +HI*(Y[i+1]-Y[i])+(H/6.0F)*(SI-SI1);
        C[i] = (HI/2.0F)*(SI*X[i+1]-SI1*X[i]);
        D[i] = (HI/6.0F)*(SI1-SI);
    }
    return;

}

void inirngG()
/*******************************************************************
c*    Set iseed1 and iseed2 for all threads with random numbers    *
c*                                                                 *
c*    Input:                                                       *
c*    Output:                                                      *
c*      iseed1 -> random number                                    *
c*      iseed2 -> random number                            *
c******************************************************************/
{
    srand( (unsigned int)time(NULL) );
    
//  generate randseed at CPU
    int *iseed1_h = (int*) malloc(sizeof(int)*NRAND);
    for(int i = 0; i < NRAND; i++)
    {
        iseed1_h[i] = rand();
    }
    int *iseed1;
    cudaMalloc((void**) &iseed1, sizeof(int)*NRAND);
//  copy to GPU 
    cudaMemcpy(iseed1, iseed1_h, sizeof(int)*NRAND, cudaMemcpyHostToDevice);
    free(iseed1_h);

    int nblocks;
    nblocks = 1 + (NRAND - 1)/NTHREAD_PER_BLOCK_GPET ;
    setupcuseed<<<nblocks, NTHREAD_PER_BLOCK_GPET>>>(iseed1);
    cudaDeviceSynchronize();
    cudaFree(iseed1);
}



void rmater(float *eminph, float *emax)
/*******************************************************************
c*    Reads material data from file                                *
c*                                                                 *
c*    Output:                                                      *
c*      fname -> input file name                                   *
c*      [Emin,Eminph,Emax] -> interval where data will be gen (eV) *
c*      refz -> total atomic no of the reference material          *
c*      refz2 -> atomic no^2 of the reference material             *
c*      refmas -> atomic weight of the reference material          *
c*      refden -> density of the reference material (g/cm^3)       *
c******************************************************************/
{
    char buffer[100];
    float shigh,slow,ecross, temp,wcc,wcb;
    //char fname[] = "data/pre4phot.matter";
    char fname[] = "data/input4gPET.matter";
    printf("rmater: Reading %s\n", fname);

    FILE *fp = fopen(fname,"r");
    FILEEXIST(fp);
    fgets(buffer, 100, fp);
    fgets(buffer, 100, fp);
    fgets(buffer, 100, fp);
    fgets(buffer, 100, fp);
    printf("%s\n",buffer);
    fscanf(fp,"%f %f %f\n",eminph, &temp, emax);
    printf("%e %e %e\n",*eminph,temp, *emax);

    fgets(buffer, 100, fp);
    //printf("%s\n",buffer);
    fscanf(fp,"%f %f\n",&wcc, &wcb);
    //printf("%e %e\n",wcc,wcb);

    fgets(buffer, 100, fp);
    //printf("%s\n",buffer);
    fscanf(fp,"%f %f %f\n",&shigh,&slow,&ecross);
    //printf("%e %e %e\n",shigh,slow,ecross);

    fgets(buffer, 100, fp);
    //printf("%s\n",buffer);
    fscanf(fp,"%d\n", &nmat_h);
    //printf("%d\n",nmat_h);
    cudaMemcpyToSymbol(nmat, &nmat_h, sizeof(int), 0, cudaMemcpyHostToDevice) ;
    if (nmat_h > MAXMAT)
    {
        printf("rmater:error: Too many materials.\n");
        exit(1);
    }

    for(int i = 0; i < nmat_h; i++)
    {
//      Read name of material, remove trailing blanks:
        float matden;
        int nelem;
        fgets(buffer,100,fp);
        //printf("%s\n", buffer);
        fgets(buffer, 100, fp);
        //printf("%s\n", buffer);
        fscanf(fp,"%f\n", &matden);
        //printf("%e\n", matden);
        fgets(buffer, 100, fp);
        //printf("%s\n",buffer);
        fscanf(fp,"%d\n",&nelem);
        //printf("%d\n", nelem);
        for(int j = 0; j < nelem; j++)
        {
            fgets(buffer, 100, fp);
            //printf("%s\n",buffer);
        }
        fgets(buffer, 100, fp);
        //printf("%s\n",buffer);
        float atnotemp,atno2temp;
        fscanf(fp,"%f %f %f\n",&atnotemp, &atno2temp, &temp);
        //printf("%e %e\n", atnotemp,atno2temp);
        fgets(buffer, 100, fp);
        //printf("%s\n",buffer);
        float mass;
        fscanf(fp,"%f\n", &mass);
        //printf("%e\n", mass);
        fgets(buffer, 100, fp);
        //printf("%s\n",buffer);
        float zmass,z2mass;
        fscanf(fp,"%f %f\n", &zmass,&z2mass);
        //printf("%e %e\n", zmass,z2mass);
    }
    fclose(fp);

    printf("\nread material: Done.\n");
}

void rlamph()
/*******************************************************************
c*    Reads photon total inverse mean free path data from file and *
c*    sets up interpolation matrices                               *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
    char buffer[100];
    int ndata;
    float dummya[NLAPH],dummyb[NLAPH],dummyc[NLAPH],dummyd[NLAPH];
    //char fname[] = "data/pre4phot.lamph";
    char fname[]="data/input4gPET.lamph";
    printf("rlamph: Reading %s\n", fname);
    FILE *fp = fopen(fname,"r");
    FILEEXIST(fp);
    fgets(buffer,100,fp);
    fgets(buffer,100,fp);
    for(int j = 0; j < nmat_h; j++)
    {
        fgets(buffer,100,fp);
        float temp;
        fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
        if (ndata != NLAPH)
        {
            printf("rlamph:error: Array dim do not match:\n");
            printf("%d %d\n", ndata,NLAPH);
            exit(1);
        }
        fgets(buffer,100,fp);
//      Preparing interpolation:
        for(int i = 0; i < NLAPH; i++)
        {
            fscanf(fp,"%f %f\n",&elaph_h[i],&lamph_h[ind2To1(j,i,MAXMAT,NLAPH)]);//excess ind2To1 equal to j*NLAPH+i,linearization
            //if(i<3)
                //printf("material %d, energy %e, cross section %e\n",j, elaph_h[i],lamph_h[ind2To1(j,i,MAXMAT,NLAPH)]);
        }
        fgets(buffer,100,fp);
        spline(elaph_h, &lamph_h[ind2To1(j,0,MAXMAT,NLAPH)],dummya,dummyb,dummyc,dummyd,0.0F,0.0F,NLAPH);
//      Loading dummy arrays into multimaterial sp matrices:
        for(int i = 0; i < NLAPH; i++)
        {
            lampha_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummya[i];
            lamphb_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummyb[i];
            lamphc_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummyc[i];
            lamphd_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummyd[i];
        }
    }
    fclose(fp);

    idleph_h = (NLAPH-1)/(elaph_h[NLAPH-1]-elaph_h[0]);
    cudaMemcpyToSymbol(idleph, &idleph_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(elaph0, &elaph_h[0], sizeof(float), 0, cudaMemcpyHostToDevice);

    cudaMallocArray(&lamph, &lamph_tex.channelDesc, NLAPH*MAXMAT, 1);
    cudaMemcpyToArray(lamph, 0, 0, lamph_h, sizeof(float)*NLAPH*MAXMAT, cudaMemcpyHostToDevice);
    lamph_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(lamph_tex, lamph);
}

void rcompt()
/*******************************************************************
c*    Reads Compton inverse mean free path data from file and sets *
c*    up interpolation matrices                                    *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
    char buffer[100];
    int ndata;
    //char fname[] = "data/pre4phot.compt";
    char fname[]= "data/input4gPET.compt";
    printf("rcompt: Reading %s\n", fname);
    FILE *fp = fopen(fname, "r");
    FILEEXIST(fp);
    fgets(buffer,100,fp);
    fgets(buffer,100,fp);
    for(int j = 0; j < nmat_h; j++)
    {
        fgets(buffer,100,fp);
        float temp;
        fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
        if (ndata != NCMPT)
        {
            printf("rcompt:error: Array dim do not match:\n");
            printf("%d %d \n", ndata,NCMPT);
            exit(1);
        }
        fgets(buffer,100,fp);
//      Preparing interpolation:
        for(int i = 0; i <NCMPT; i++)
        {
            fscanf(fp,"%f %f\n",&ecmpt_h[i],&compt_h[ind2To1(j,i,MAXMAT,NCMPT)]);
//                      if(j == nmat-1)
//                              printf("%e %e\n",ecmpt[i],compt[i]);
        }
        fgets(buffer,100,fp);

    }
    fclose(fp);

    idlecp_h = (NCMPT-1)/(ecmpt_h[NCMPT-1]-ecmpt_h[0]);
    cudaMemcpyToSymbol(idlecp, &idlecp_h, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ecmpt0, &ecmpt_h[0], sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaMallocArray(&compt, &compt_tex.channelDesc, NCMPT*MAXMAT, 1);
    cudaMemcpyToArray(compt, 0, 0, compt_h, sizeof(float)*NCMPT*MAXMAT, cudaMemcpyHostToDevice);
    compt_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(compt_tex, compt);
}

void rcmpsf()
/*******************************************************************
c*    Reads Compton scattering function data from file and         *
c*    sets up interpolation matrices                               *
c******************************************************************/
{
    char buffer[100];
    //char fname[] = "data/pre4phot.cmpsf";
    char fname[]= "data/input4gPET.cmpsf";
    printf("rcmpsf: Reading %s\n", fname);
    FILE *fp = fopen(fname,"r");
    fgets(buffer,100,fp);
    fgets(buffer,100,fp);
    for(int j = 0; j < nmat_h; j++)
    {
//  read sf data
        fgets(buffer,100,fp);
        float temp;
        int ndata;
        fscanf(fp,"%d %f %f %f\n",&ndata,&temp,&temp,&temp);
        fgets(buffer,100,fp);
        for(int i = 0; i < ndata; i++)
        {
            fscanf(fp,"%f %f %f\n",&temp, &temp, &temp);
        }

//  read s surface
        fgets(buffer,100,fp);
        int ncp, ne;
        float dcp, de;
        fscanf(fp,"%d %f %f %f %d %f %f %f\n", &ncp, &temp, &temp, &dcp, &ne, &temp, &temp, &de);
        if (ncp != NCPCM)
        {
            printf("rcmpsf:error: NCP dim do not match:\n");
            printf("%d %d\n", ncp,NCPCM);
            exit(1);
        }
        if (ne != NECM)
        {
            printf("rcmpsf:error: NE dim do not match:\n");
            printf("%d %d\n", ne,NECM);
            exit(1);
        }
        idcpcm_h = 1.0f/dcp;
        idecm_h = 1.0f/de;
        for(int icp=0; icp <ncp; icp++)
            fscanf(fp,"%f ",&temp);
        fscanf(fp,"\n");
        for(int ie=0; ie <ne; ie++)
            fscanf(fp,"%f ",&temp);
        fscanf(fp,"\n");
        for(int icp=0; icp <ncp; icp++)
        {
            for(int ie = 0; ie<ne; ie++)
            {
                fscanf(fp,"%f ",&mucmpt_h[j*NCPCM*NECM+icp*NECM+ie]);
//              if(mucmpt_h[j*NCPCM*NECM+icp*NECM+ie] > 1.0f || mucmpt_h[j*NCPCM*NECM+icp*NECM+ie]<-1.0f)
//                  cout << "error in data" << mucmpt_h[j*NCPCM*NECM+icp*NECM+ie] << endl;
            }
            fscanf(fp,"\n");
        }
        fscanf(fp,"\n");
    }
    fclose(fp);

//  load to GPU
    cudaMemcpyToSymbol(idcpcm, &idcpcm_h, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(idecm, &idecm_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    const cudaExtent volumeSize = make_cudaExtent(NECM, NCPCM, MAXMAT);

    cudaMalloc3DArray(&sArray, &channelDesc, volumeSize) ;
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)mucmpt_h, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = sArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams) ;

    s_tex.normalized = false;
    s_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(s_tex, sArray, channelDesc);
}

void rphote()
/*******************************************************************
c*    Reads photoelectric inverse mean free path data from file and*
c*    sets up interpolation matrices                               *
c******************************************************************/
{
    char buffer[100];
    int ndata;
    //char fname[] = "data/pre4phot.phote";
    char fname[]= "data/input4gPET.phote";
    printf("rphote: Reading %s\n", fname);
    FILE *fp = fopen(fname,"r");
    fgets(buffer,100,fp);
    fgets(buffer,100,fp);
    for(int j = 0; j < nmat_h; j++)
    {
        fgets(buffer,100,fp);
        float temp;
        fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
        if (ndata != NPHTE)
        {
            printf("rphote:error: Array dim do not match:\n");
            printf("%d %d\n", ndata,NPHTE);
            exit(1);
        }
        fgets(buffer,100,fp);
//      Preparing interpolation
        for(int i = 0; i < NPHTE; i++)
        {
            fscanf(fp,"%f %f\n",&ephte_h[i],&phote_h[ind2To1(j,i,MAXMAT,NPHTE)]);
//                      if(j == nmat-1)
//                              printf("%e %e\n",ephte[i],phote[i]);
        }
        fgets(buffer,100,fp);
    }
    fclose(fp);

    idlepe_h = (NPHTE-1)/(ephte_h[NPHTE-1]-ephte_h[0]);
    cudaMemcpyToSymbol(idlepe, &idlepe_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(ephte0, &ephte_h[0], sizeof(float), 0, cudaMemcpyHostToDevice);

    cudaMallocArray(&phote, &phote_tex.channelDesc, NPHTE*MAXMAT, 1);
    cudaMemcpyToArray(phote, 0, 0, phote_h, sizeof(float)*NPHTE*MAXMAT, cudaMemcpyHostToDevice);
    phote_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(phote_tex, phote);

}
void rrayle()
/*******************************************************************
c*    Reads rayleigh inverse mean free path data from file and     *
c*    sets up interpolation matrices                               *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
    char buffer[100];
    int ndata;
    //char fname[] = "data/pre4phot.rayle";
    char fname[]="data/input4gPET.rayle";
    printf("rrayle: Reading %s\n", fname);
    FILE *fp = fopen(fname,"r");
    fgets(buffer,100,fp);
    fgets(buffer,100,fp);
    for(int j = 0; j < nmat_h; j++)
    {
        fgets(buffer,100,fp);
        float temp;
        fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
        if (ndata != NRAYL)
        {
            printf("rrayle:error: Array dim do not match:\n");
            printf("%d %d\n", ndata,NRAYL);
            exit(1);
        }
        fgets(buffer,100,fp);
//      Preparing interpolation
        for(int i = 0; i < NRAYL; i++)
        {
            fscanf(fp,"%f %f\n",&erayl_h[i],&rayle_h[ind2To1(j,i,MAXMAT,NRAYL)]);
        }
        fgets(buffer,100,fp);
    }
    fclose(fp);

    idlerl_h = (NRAYL-1)/(erayl_h[NRAYL-1]-erayl_h[0]);
    cudaMemcpyToSymbol(idlerl, &idlerl_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(erayl0, &erayl_h[0], sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaMallocArray(&rayle, &rayle_tex.channelDesc, NRAYL*MAXMAT, 1);
    cudaMemcpyToArray(rayle, 0, 0, rayle_h, sizeof(float)*NRAYL*MAXMAT, cudaMemcpyHostToDevice);
    rayle_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(rayle_tex, rayle);

}

void rrayff()
/*******************************************************************
c*    Reads Rayleigh scattering form factor data from file and     *
c*    sets up interpolation matrices                               *
c******************************************************************/
{
    char buffer[100];
    //char fname[] = "data/pre4phot.rayff";
    char fname[]= "data/input4gPET.rayff";
    printf("rrayff: Reading %s\n", fname);
    FILE *fp = fopen(fname,"r");
    fgets(buffer,100,fp);
    fgets(buffer,100,fp);
    for(int j = 0; j < nmat_h; j++)
    {
//  read ff data
        fgets(buffer,100,fp);
        float temp;
        int ndata;
        fscanf(fp,"%d %f %f %f\n",&ndata,&temp,&temp,&temp);
        fgets(buffer,100,fp);
        for(int i = 0; i < ndata; i++)
        {
            fscanf(fp,"%f %f  %f\n",&temp, &temp, &temp);
        }

//  read f surface
        fgets(buffer,100,fp);
        int ncp, ne;
        float dcp, de;
        fscanf(fp,"%d %f %f %f %d %f %f %f\n", &ncp, &temp, &temp, &dcp, &ne, &temp, &temp, &de);
        if (ncp != NCPRL)
        {
            printf("rrayff:error: NCP dim do not match:\n");
            printf("%d %d\n", ncp,NCPRL);
            exit(1);
        }
        if (ne != NERL)
        {
            printf("rrayff:error: NE dim do not match:\n");
            printf("%d %d\n", ne,NERL);
            exit(1);
        }
        idcprl_h = 1.0f/dcp;
        iderl_h = 1.0f/de;
        for(int icp=0; icp <ncp; icp++)
            fscanf(fp,"%f ",&temp);
        fscanf(fp,"\n");
        for(int ie=0; ie <ne; ie++)
            fscanf(fp,"%f ",&temp);
        fscanf(fp,"\n");
        for(int icp=0; icp <ncp; icp++)
        {
            for(int ie = 0; ie<ne; ie++)
            {
                fscanf(fp,"%f ",&murayl_h[j*NCPRL*NERL+icp*NERL+ie]);
//                                  if(murayl_h[j*NCPRL*NERL+icp*NERL+ie] > 1.0f || murayl_h[j*NCPRL*NERL+icp*NERL+ie]<-1.0f)
//                                      cout << "error in data" << murayl_h[j*NCPRL*NERL+icp*NERL+ie] << endl;
            }
            fscanf(fp,"\n");
        }
        fscanf(fp,"\n");
//      cout << murayl_h[j*NCPRL*NERL+(NCPRL-2)*NERL+1] << endl;
    }
    fclose(fp);

//  load to GPU
    cudaMemcpyToSymbol(idcprl, &idcprl_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(iderl, &iderl_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    const cudaExtent volumeSize = make_cudaExtent(NERL, NCPRL, MAXMAT);

    cudaMalloc3DArray(&fArray, &channelDesc, volumeSize) ;
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)murayl_h, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = fArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    f_tex.normalized = false;
    f_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(f_tex, fArray, channelDesc);
}



float itphip(int matid, float e)
/*******************************************************************
c*    Photon total inverse mean free path --3spline interpolation  *
c*                                                                 *
c*    Input:                                                       *
c*      matid -> material id#                                      *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      Total inverse mean free path in cm^2/g                     *
c******************************************************************/
{
    int i;

    i = int(idleph_h*(e-elaph_h[0]));

    return  lampha_h[ind2To1(matid,i,MAXMAT,NLAPH)]
            + e*(lamphb_h[ind2To1(matid,i,MAXMAT,NLAPH)]
                 + e*(lamphc_h[ind2To1(matid,i,MAXMAT,NLAPH)]
                      + e*lamphd_h[ind2To1(matid,i,MAXMAT,NLAPH)] ));
}

void iniwck(float eminph,float emax, Phantom phantom)
/*******************************************************************
c*    Finds information used to transport photons with the Woodcock*
c*    technique                                                    *
c*                                                                 *
c*    Input:                                                       *
c*      eminph -> minimum photon energy in data files (eV)         *
c*      emax -> maximum photon energy in data files (eV)           *
c*    Output                                                       *
c*      bytes -> space allocated for arrays                        *
c*    Comments:                                                    *
c*      -> common /dpmsrc/ must be loaded previously               *
c*      -> rlamph() must be called previously                      *
c*      -> emax reduced to avoid reaching the end of interpol table*
c******************************************************************/
{
    float maxden[MAXMAT],de,e,ymax,ycanbe;
    const float eps = 1.0e-10F;
    unsigned int NXYZ = phantom.Unxvox*phantom.Unyvox*phantom.Unzvox;
    printf("iniwck phantom: Started.\n");
//      Find the largest density for each present material:
    for(int i = 0; i < MAXMAT; i++)
    {
        maxden[i] = 0.0F;
    }
    for(int vox = 0; vox < NXYZ; vox++)
    {
        if (phantom.dens[vox] > maxden[phantom.mat[vox]])
            maxden[phantom.mat[vox]] = phantom.dens[vox];
    }
    
//      Prepare data:
    wcke0_h = eminph;
    de = (emax*(1.0F - eps ) - wcke0_h ) / NWCK;
    idlewk_h = 1.0F/de;

    for(int i = 0; i < NWCK; i++)
    {
        e = wcke0_h + de*i;
        ymax = 0.0;
        for(int j = 0; j < nmat_h; j++)
        {
            ycanbe = itphip(j,e)*maxden[j];

            if (ycanbe > ymax) ymax = ycanbe;
        }
        woock_h[i] = 1.0F/ymax;
    }

    cudaMemcpyToSymbol(idlewk, &idlewk_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(wcke0, &wcke0_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaMallocArray(&woock, &woock_tex.channelDesc, NWCK, 1);
    cudaMemcpyToArray(woock, 0, 0, woock_h, sizeof(float)*NWCK, cudaMemcpyHostToDevice);
    woock_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(woock_tex, woock);
}

void initPhantom(Phantom phantom)
{
    printf("CT dimension: %d %d %d\n", phantom.Unxvox, phantom.Unyvox, phantom.Unzvox);
    printf("CT resolution: %f %f %f\n", phantom.dx, phantom.dy, phantom.dz);

    cudaMemcpyToSymbol(Unxvox, &phantom.Unxvox, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Unyvox, &phantom.Unyvox, sizeof(int), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(Unzvox, &phantom.Unzvox, sizeof(int), 0, cudaMemcpyHostToDevice) ;


    cudaMemcpyToSymbol(dx_gBrachy, &phantom.dx, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(dy_gBrachy, &phantom.dy, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(dz_gBrachy, &phantom.dz, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    float idx_gBrachy_h,idy_gBrachy_h,idz_gBrachy_h;
    idx_gBrachy_h = 1.0F/phantom.dx;
    cudaMemcpyToSymbol(idx_gBrachy, &idx_gBrachy_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    idy_gBrachy_h = 1.0F/phantom.dy;
    cudaMemcpyToSymbol(idy_gBrachy, &idy_gBrachy_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    idz_gBrachy_h = 1.0F/phantom.dz;
    cudaMemcpyToSymbol(idz_gBrachy, &idz_gBrachy_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaMemcpyToSymbol(Offsetx_gBrachy, &phantom.Offsetx, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(Offsety_gBrachy, &phantom.Offsety, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(Offsetz_gBrachy, &phantom.Offsetz, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaExtent volumeSize = make_cudaExtent(phantom.Unxvox, phantom.Unyvox, phantom.Unzvox);
    CUDA_CALL(cudaMalloc3DArray(&mat, &mat_tex.channelDesc, volumeSize));
    CUDA_CALL(cudaMalloc3DArray(&dens, &dens_tex.channelDesc, volumeSize));

//      create a 3d array on device
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)phantom.mat, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
    copyParams.dstArray = mat;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams) ;
//      copy data from host to device
    mat_tex.normalized = false;
    mat_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(mat_tex, mat, mat_tex.channelDesc);
//      bind to texture memory

    copyParams.srcPtr   = make_cudaPitchedPtr((void*)phantom.dens, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    copyParams.dstArray = dens;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams) ;
//      copy data from host to device
    dens_tex.normalized = false;
    dens_tex.filterMode = cudaFilterModePoint;
    cudaBindTextureToArray(dens_tex, dens, dens_tex.channelDesc);
//      bind to texture memory
}//*/

void init(Phantom phantom)
/*******************************************************************
c*    Initializes the gCTD system                                  *
c******************************************************************/
{
    initPhantom(phantom);    
    cudaMemcpyToSymbol(eabsph, &eabsph_h, sizeof(float), 0, cudaMemcpyHostToDevice);
//      in GPU, initialize rand seed with rand numbers
    inirngG();

    rmater(&eminph, &emax);//no use?
    printf("\n");
    if(eabsph_h <eminph)
    {
        printf("init:error: Eabs out of range.\n");
        exit(1);
    }

//  load total cross section
    rlamph();
//  load compton cross section
    rcompt();
    rcmpsf();
//  load photoelectric cross section
    rphote();
//      load rayleigh cross section and form factors
    rrayle();
    rrayff();
//      iniwck must be called after reading esrc & eabsph:
    iniwck(eminph, emax, phantom);
    printf("\n\nInitialize : Done.\n");//*/
}

void iniwck(float eminph,float emax, struct object_v* objectMaterial) //for detector
{
    float maxden[MAXMAT],de,e,ymax,ycanbe;
    const float eps = 1.0e-10F;

    printf("\n");
    printf("\n");
    printf("iniwck detector: Started.\n");
//  Find the largest density for each present material:
    for(int i = 0; i < MAXMAT; i++)
    {
        maxden[i] = 0.0F;
    }
    for(int i=0; i<2; i++)
    {
        if (objectMaterial[i].density > maxden[objectMaterial[i].material])
            maxden[objectMaterial[i].material] = objectMaterial[i].density;
    }

        
//  Prepare data:
    wcke0_h = eminph;
    de = (emax*(1.0F - eps ) - wcke0_h ) / NWCK;
    idlewk_h = 1.0F/de;

    for(int i = 0; i < NWCK; i++)
    {
        e = wcke0_h + de*i;
        ymax = 0.0;
        for(int j = 0; j < nmat_h; j++)
        {
            ycanbe = itphip(j,e)*maxden[j];
            if (ycanbe > ymax) 
                ymax = ycanbe;
        }
        woock_h[i] = 1.0F/ymax;
        /*if (i<1100 && i>1090)
            printf("1/lamda=%f\n",woock_h[i]);*/
    }

    cudaMemcpyToSymbol(idlewk, &idlewk_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(wcke0, &wcke0_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaMallocArray(&woockde, &woockde_tex.channelDesc, NWCK, 1);
    cudaMemcpyToArray(woockde, 0, 0, woock_h, sizeof(float)*NWCK, cudaMemcpyHostToDevice);
    woockde_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(woockde_tex, woockde);
}


void iniPanel(struct object_t* objectArray, struct object_v* objectMaterial,int totalOb)
/*******************************************************************
c*    Initializes the module system                                  *
c******************************************************************/
{

    printf(" \n");
    printf("init: Panel geometry;\n");

//  copy arrays from host to device
    int *ma=new int[2];
    float *den=new float[2];
    int *p_id=new int[totalOb];

    float *lx_m=new float[totalOb];
    float *ly_m=new float[totalOb];
    float *lz_m=new float[totalOb];

    float *Mx_m=new float[totalOb];
    float *My_m=new float[totalOb];
    float *Mz_m=new float[totalOb];

    float *Msx_m=new float[totalOb];
    float *Msy_m=new float[totalOb];
    float *Msz_m=new float[totalOb];

    float *Lx_m=new float[totalOb];
    float *Ly_m=new float[totalOb];
    float *Lz_m=new float[totalOb];

    float *sx_m=new float[totalOb];
    float *sy_m=new float[totalOb];
    float *sz_m=new float[totalOb];

    float *ox_m=new float[totalOb];
    float *oy_m=new float[totalOb];
    float *oz_m=new float[totalOb];

    float *dx_m=new float[totalOb];
    float *dy_m=new float[totalOb];
    float *dz_m=new float[totalOb];

    float *UXx_m=new float[totalOb];
    float *UXy_m=new float[totalOb];
    float *UXz_m=new float[totalOb];

    float *UYx_m=new float[totalOb];
    float *UYy_m=new float[totalOb];
    float *UYz_m=new float[totalOb];

    float *UZx_m=new float[totalOb];
    float *UZy_m=new float[totalOb];
    float *UZz_m=new float[totalOb];

    for (int i=0;i<2;i++)
    {
        ma[i]=objectMaterial[i].material;
        den[i]=objectMaterial[i].density;
    }

    for (int i=0;i<totalOb;i++)
    {
        p_id[i]=objectArray[i].panel;

        lx_m[i]=objectArray[i].lengthx;
        ly_m[i]=objectArray[i].lengthy;
        lz_m[i]=objectArray[i].lengthz;

        Mx_m[i]=objectArray[i].MODx;
        My_m[i]=objectArray[i].MODy;
        Mz_m[i]=objectArray[i].MODz;

        Msx_m[i]=objectArray[i].Mspacex;
        Msy_m[i]=objectArray[i].Mspacey;
        Msz_m[i]=objectArray[i].Mspacez;

        Lx_m[i]=objectArray[i].LSOx;
        Ly_m[i]=objectArray[i].LSOy;
        Lz_m[i]=objectArray[i].LSOz;

        sx_m[i]=objectArray[i].spacex;
        sy_m[i]=objectArray[i].spacey;
        sz_m[i]=objectArray[i].spacez;

        ox_m[i]=objectArray[i].offsetx;
        oy_m[i]=objectArray[i].offsety;
        oz_m[i]=objectArray[i].offsetz;

        dx_m[i]=objectArray[i].directionx;
        dy_m[i]=objectArray[i].directiony;
        dz_m[i]=objectArray[i].directionz;

        UXx_m[i]=objectArray[i].UniXx;
        UXy_m[i]=objectArray[i].UniXy;
        UXz_m[i]=objectArray[i].UniXz;

        UYx_m[i]=objectArray[i].UniYx;
        UYy_m[i]=objectArray[i].UniYy;
        UYz_m[i]=objectArray[i].UniYz;

        UZx_m[i]=objectArray[i].UniZx;
        UZy_m[i]=objectArray[i].UniZy;
        UZz_m[i]=objectArray[i].UniZz; 
        
    }
    int Mn, Ln;
    Mn=floorf(ly_m[0]/(My_m[0]+Msy_m[0]))+1;    
    Ln=floorf(My_m[0]/(Ly_m[0]+sy_m[0]))+1;
    //printf("Mn %d Ln %d\n", Mn, Ln);
    cudaMemcpyToSymbol(crystalNy, &Ln, sizeof(int));
    cudaMemcpyToSymbol(moduleNy, &Mn, sizeof(int));

    Mn*=floorf(lz_m[0]/(Mz_m[0]+Msz_m[0]))+1;
    Ln*=floorf(Mz_m[0]/(Lz_m[0]+sz_m[0]))+1;
    //printf("Mn %d Ln %d\n", Mn, Ln);

    cudaMemcpyToSymbol(crystalN, &Ln, sizeof(int));
    cudaMemcpyToSymbol(moduleN, &Mn, sizeof(int));
    
    
    cudaMemcpyToSymbol(dev_totalPanels, &totalOb, sizeof(int), 0, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&mat_panel, 2 * sizeof(int)); 
    cudaMalloc((void**)&dens_panel, 2 * sizeof(float)); 
    cudaMalloc((void**)&panelID, totalOb * sizeof(int)); 

    cudaMalloc((void**)&lengthx_panel, totalOb * sizeof(float)); 
    cudaMalloc((void**)&lengthy_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&lengthz_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&MODx_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&MODy_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&MODz_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&Mspacex_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&Mspacey_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&Mspacez_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&LSOx_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&LSOy_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&LSOz_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&spacex_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&spacey_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&spacez_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&offsetx_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&offsety_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&offsetz_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&directionx_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&directiony_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&directionz_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&UniXx_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&UniXy_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&UniXz_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&UniYx_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&UniYy_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&UniYz_panel, totalOb * sizeof(float));

    cudaMalloc((void**)&UniZx_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&UniZy_panel, totalOb * sizeof(float));
    cudaMalloc((void**)&UniZz_panel, totalOb * sizeof(float));

    cudaMemcpy(mat_panel, ma, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dens_panel, den, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(panelID, p_id, totalOb*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(lengthx_panel, lx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lengthy_panel, ly_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(lengthz_panel, lz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(MODx_panel, Mx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(MODy_panel, My_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(MODz_panel, Mz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(Mspacex_panel, Msx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Mspacey_panel, Msy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Mspacez_panel, Msz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(LSOx_panel, Lx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(LSOy_panel, Ly_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(LSOz_panel, Lz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(spacex_panel, sx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(spacey_panel, sy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(spacez_panel, sz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(offsetx_panel, ox_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(offsety_panel, oy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(offsetz_panel, oz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(directionx_panel, dx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(directiony_panel, dy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(directionz_panel, dz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(UniXx_panel, UXx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(UniXy_panel, UXy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(UniXz_panel, UXz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(UniYx_panel, UYx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(UniYy_panel, UYy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(UniYz_panel, UYz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(UniZx_panel, UZx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(UniZy_panel, UZy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(UniZz_panel, UZz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

    delete[] ma;
    delete[] den;
    delete[] p_id;

    delete[] lx_m;
    delete[] ly_m;
    delete[] lz_m;

    delete[] Mx_m;
    delete[] My_m;
    delete[] Mz_m;

    delete[] Msx_m;
    delete[] Msy_m;
    delete[] Msz_m;

    delete[] Lx_m;
    delete[] Ly_m;
    delete[] Lz_m;

    delete[] sx_m;
    delete[] sy_m;
    delete[] sz_m;

    delete[] ox_m;
    delete[] oy_m;
    delete[] oz_m;

    delete[] dx_m;
    delete[] dy_m;
    delete[] dz_m;

    delete[] UXx_m;
    delete[] UXy_m;
    delete[] UXz_m;

    delete[] UYx_m;
    delete[] UYy_m;
    delete[] UYz_m;

    delete[] UZx_m;
    delete[] UZy_m;
    delete[] UZz_m;
    
    printf("finish init: Module geometry;\n\n");
}

#endif
