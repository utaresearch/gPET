#ifndef __GPETKERNAL__
#define __GPETKERNAL__

#include "gPET.h"
#include "gPET_kernals.h"

__global__ void setupcuseed(int* iseed1)
//Setup random seeds, used for random sampling in GPU code
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
//      obtain current id on thread
   if( id < NRAND)
    {
        curand_init(iseed1[id], id, 0, &cuseed[id]);
        if(id<5) printf("first 5 cuseeds are %u \n",cuseed[id]);
    }
}

__device__ int4 getAbsVox(float3 xtemp)
//Return the absolute vox index according to the coordinate
{
    int4 temp;
    temp.z = (xtemp.z-Offsetz_gBrachy)*idz_gBrachy;
    temp.y = (xtemp.y-Offsety_gBrachy)*idy_gBrachy;
    temp.x = (xtemp.x-Offsetx_gBrachy)*idx_gBrachy;
//The following give the boundry condition
    temp.w = (temp.x <= 0.0f || temp.x >= Unxvox || temp.y <= 0.0f || temp.y >= Unyvox || temp.z <= 0.0f || temp.z >= Unzvox)?-1 : 1;
    return temp;
}

__device__ float lamwck(float e)
//Minimum Mean free path prepared to play the Woodcock trick for materials in phantom
{
    float i = idlewk*(e-wcke0) + 0.5f;
    return tex1D(woock_tex, i);
}

__device__ float lamwckde(float e)
//Minimum Mean free path prepared to play the Woodcock trick for materials in detector
{
    float i = idlewk*(e-wcke0) + 0.5f;
    return tex1D(woockde_tex, i);
}

__device__ float itphip_G(int matid, float e)
//Photon total inverse mean free path 
{
    float i = idleph*(e-elaph0) + 0.5;
    return tex1D(lamph_tex,matid * NLAPH + i);
}

__device__ float irylip(int matid, float e)
//Inverse Rayleigh mean free path  
{
    float i = idlerl*(e - erayl0) + 0.5;
    return tex1D(rayle_tex,matid * NRAYL + i);
}

__device__ float icptip(int matid, float e)
//Inverse Compton mean free path  
{
    float i = idlecp*(e - ecmpt0) + 0.5;
    return tex1D(compt_tex,matid * NCMPT + i);
}

__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe, int matid)
//this is the KN model crosssection from table
/*******************************************************************
c*    Samples a Compton event following Klein-Nishina DCS          *
c*                                                                 *
c*    Input:                                                       *
c*      energy -> photon energy in eV                              *
c*    Output:                                                      *
c*      efrac -> fraction of initial energy kept by 2nd photon     *
c*      costhe -> cos(theta) of the 2nd photon                     *
c*    Comments:                                                    *
c*      -> inirng() must be called before 1st call                 *
c******************************************************************/
{
    float indcp = curand_uniform(localState_pt)*idcpcm;
    float inde = energytemp*idecm;
    float temp = tex3D(s_tex,inde+0.5f, indcp+0.5f, matid+0.5f);
    if(temp > 1.0f) temp = 1.0f;
    if(temp < -1.0f) temp = -1.0f;
    *costhe = temp;

    *efrac = 1.0f/(1.0f + energytemp*IMC2*(1.0f-temp));
}

__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe)
//this is the standard KN model which treats electron as free and no dopler effect
/*******************************************************************
c*    Samples a Compton event following Klein-Nishina DCS          *
c*                                                                 *
c*    Input:                                                       *
c*      energy -> photon energy in eV                              *
c*    Output:                                                      *
c*      efrac -> fraction of initial energy kept by 2nd photon     *
c*      costhe -> cos(theta) of the 2nd photon                     *
c*    Comments:                                                    *
c*      -> inirng() must be called before 1st call                 *
c******************************************************************/
{
    float e0,twoe,kmin2,loge,mess;

    e0 = energytemp*IMC2;
    twoe = 2.0*e0;
    kmin2 = 1.0/((1.0+twoe)*(1.0+twoe));
    loge = __logf(1.0+twoe);

    for(;;)
    {
        if (curand_uniform(localState_pt)*(loge+twoe*(1.0+e0)*kmin2) < loge)
        {
            *efrac = expf(-curand_uniform(localState_pt)*loge);
        }
        else
        {
            *efrac = sqrtf(kmin2+curand_uniform(localState_pt)*(1.0-kmin2));
        }
        mess = e0*e0*(*efrac)*(1.0+(*efrac)*(*efrac));
        if (curand_uniform(localState_pt)*mess <= mess-(1.0-*efrac)*((1.0+twoe)*(*efrac)-1.0))break;
    }

    *costhe = 1.0-(1.0-*efrac)/((*efrac)*e0);
}

__device__ void rylsam(float energytemp, int matid, curandState *localState_pt, float *costhe)
/*******************************************************************
c*    Samples a Rayleigh event following its DCS                   *
c*                                                                 *
c*    Input:                                                       *
c*      energy -> photon energy in eV                              *
c*    Output:                                                      *
c*      costhe -> cos(theta) of the 2nd photon                     *
c*    Comments:                                                    *
c*      -> inirng() must be called before 1st call                 *
c******************************************************************/
{
    float indcp = curand_uniform(localState_pt)*idcprl;
    float inde = energytemp*iderl;
    float temp = tex3D(f_tex,inde+0.5f, indcp+0.5f, matid+0.5f);
    if(temp > 1.0f) temp = 1.0f;
    if(temp < -1.0f) temp = -1.0f;
    *costhe = temp;
}

__device__ float getDistance(float3 coords, float4 direcs)
//special case for sphere, need modification dor other cases
/*******************************************************************
c*   get the distance to the recording plane           *
c*      distance-> nearest distance to current body boundaries     *
c******************************************************************/
{
    coords.x=coords.x-recordsphere_d[0];
    coords.y=coords.y-recordsphere_d[1];
    coords.z=coords.z-recordsphere_d[2];
    float t;
    float a= direcs.x * direcs.x + direcs.y * direcs.y + direcs.z * direcs.z;
    float b = 2.0f*(direcs.x * coords.x + direcs.y * coords.y + direcs.z * coords.z); 
    float c = (coords.x * coords.x + coords.y * coords.y +  coords.z * coords.z)-(recordsphere_d[3])*(recordsphere_d[3]);
    if(b*b-4*a*c<0) return 0;
    if(c<0)
        t=(-b+sqrtf(b*b-4*a*c))/(2*a);
    else if(b<0)
        t=(-b-sqrtf(b*b-4*a*c))/(2*a);
    else
        t=(-b+sqrtf(b*b-4*a*c))/(2*a);

    return t;//direction normalized
}
__device__ void rotate(float *u, float *v, float *w, float costh, float phi)
/*******************************************************************
c*    Rotates a vector; the rotation is specified by giving        *
c*    the polar and azimuthal angles in the "self-frame", as       *
c*    determined by the vector to be rotated.                      *
c*                                                                 *
c*    Input:                                                       *
c*      (u,v,w) -> input vector (=d) in the lab. frame             *
c*      costh -> cos(theta), angle between d before and after turn *
c*      phi -> azimuthal angle (rad) turned by d in its self-frame *
c*    Output:                                                      *
c*      (u,v,w) -> rotated vector components in the lab. frame     *
c*    Comments:                                                    *
c*      -> (u,v,w) should have norm=1 on input; if not, it is      *
c*         renormalized on output, provided norm>0.                *
c*      -> The algorithm is based on considering the turned vector *
c*         d' expressed in the self-frame S',                      *
c*           d' = (sin(th)cos(ph), sin(th)sin(ph), cos(th))        *
c*         and then apply a change of frame from S' to the lab     *
c*         frame. S' is defined as having its z' axis coincident   *
c*         with d, its y' axis perpendicular to z and z' and its   *
c*         x' axis equal to y'*z'. The matrix of the change is then*
c*                   / uv/rho    -v/rho    u \                     *
c*          S ->lab: | vw/rho     u/rho    v |  , rho=(u^2+v^2)^0.5*
c*                   \ -rho       0        w /                     *
c*      -> When rho=0 (w=1 or -1) z and z' are parallel and the y' *
c*         axis cannot be defined in this way. Instead y' is set to*
c*         y and therefore either x'=x (if w=1) or x'=-x (w=-1)    *
c******************************************************************/
{
    float rho2,sinphi,cosphi,sthrho,urho,vrho,sinth,norm;

    rho2 = (*u)*(*u)+(*v)*(*v);
    norm = rho2 + (*w)*(*w);
//      Check normalization:
    if (fabs(norm-1.0) > SZERO)
    {
//      Renormalize:
        norm = 1.0/__fsqrt_rn(norm);
        *u = (*u)*norm;
        *v = (*v)*norm;
        *w = (*w)*norm;
    }

    sinphi = __sinf(phi);
    cosphi = __cosf(phi);
//      Case z' not= z:

    float temp = costh*costh;
    if (rho2 > ZERO)
    {
        if(temp < 1.0f)
            sthrho = __fsqrt_rn((1.00-temp)/rho2);
        else
            sthrho = 0.0f;

        urho =  (*u)*sthrho;
        vrho =  (*v)*sthrho;
        *u = (*u)*costh - vrho*sinphi + (*w)*urho*cosphi;
        *v = (*v)*costh + urho*sinphi + (*w)*vrho*cosphi;
        *w = (*w)*costh - rho2*sthrho*cosphi;
    }
    else
//      2 especial cases when z'=z or z'=-z:
    {
        if(temp < 1.0f)
            sinth = __fsqrt_rn(1.00-temp);
        else
            sinth = 0.0f;

        *v = sinth*sinphi;
        if (*w > 0.0)
        {
            *u = sinth*cosphi;
            *w = costh;
        }
        else
        {
            *u = -sinth*cosphi;
            *w = -costh;
        }
    }
}

__global__ void photon(const int nactive)
/*******************************************************************
c*    Transports a photon until it either escapes from the         *
c*    phantom or its energy drops below EabsPhoton                *
c******************************************************************/
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
//      obtain current id on thread
    const double spe=29979.2458; //cm/us
    curandState localState = cuseed[id%NRAND];
    while( id < nactive)
    {
        float3 xtemp = x_gBrachy[id];
        float4 vxtemp = vx_gBrachy[id];
        double tof = d_time[id];
        //if(id <5 ) printf("x=%f,y=%f,z=%f,t = %f, vx=%f,vy=%f,vz=%f,e=%f\n",xtemp.x,xtemp.y,xtemp.z,tof,vxtemp.x,vxtemp.y,vxtemp.z,vxtemp.w);
        if(vxtemp.w<0||tof<=0) {id+=blockDim.x*gridDim.x;continue;}
        //      Loop until it either escapes or is absorbed:
        while(1)
        {
//      Get lambda from the minimum lambda at the current energy:
            float lammin = lamwck(vxtemp.w);
            float s = -lammin*__logf(curand_uniform(&localState));
            xtemp.x += s*vxtemp.x;
            xtemp.y += s*vxtemp.y;
            xtemp.z += s*vxtemp.z;
            //xtemp.w += s/spe;
            tof += s/spe;
            int4 absvoxtemp = getAbsVox(xtemp);
            //if(id <5 ) printf("id %d absvoxtem.w=%d, x=%f,y=%f,z=%f,vx=%f,vy=%f,vz=%f,s=%f\n",id, absvoxtemp.w, xtemp.x,xtemp.y,xtemp.z,vxtemp.x,vxtemp.y,vxtemp.z,s);
            if (absvoxtemp.w == -1)//means the particle is outside the phantom
            {
#if RECORDPSF==-1
                float r=getDistance(xtemp,vxtemp);
                xtemp.x +=r*vxtemp.x;
                xtemp.y +=r*vxtemp.y;
                xtemp.z +=r*vxtemp.z;
                tof += r/spe;
#endif
                break;
            }

//  get density
            float voxden = tex3D(dens_tex, absvoxtemp.x, absvoxtemp.y, absvoxtemp.z);
//  get mat id
            int voxmatid = tex3D(mat_tex, absvoxtemp.x, absvoxtemp.y, absvoxtemp.z);

//      Apply Woodcock trick:
            float lamden = lammin*voxden;
            float prob = 1.0-lamden*itphip_G(voxmatid, vxtemp.w);

            float randno = curand_uniform(&localState);
//      No real event; continue jumping:
            if (randno < prob) continue;
//      Compton:
            prob += lamden*icptip(voxmatid, vxtemp.w);
            if (randno < prob)
            {
                float efrac, costhe;
                comsam(vxtemp.w, &localState, &efrac, &costhe, voxmatid);
//              comsam(vxtemp.w, &localState, &efrac, &costhe);
                float phi = TWOPI*curand_uniform(&localState);
                vxtemp.w *= efrac;
                if (vxtemp.w < eabsph)
                    break;

                rotate(&vxtemp.x,&vxtemp.y,&vxtemp.z,costhe,phi);
                continue;
            }
//  Rayleigh:
            prob += lamden*irylip(voxmatid, vxtemp.w);
            if (randno < prob)
            {
                float costhe;
                rylsam(vxtemp.w, voxmatid, &localState, &costhe);
                float phi = TWOPI*curand_uniform(&localState);
                rotate(&vxtemp.x,&vxtemp.y,&vxtemp.z,costhe,phi);
                continue;
            }
//  Photoelectric:
            tof = -0.5f;// t<0 means dead in phantom, won't be simulated any more
            break;
        }
        x_gBrachy[id] = xtemp;
        d_time[id] = tof;
        vx_gBrachy[id] = vxtemp;
        id+=blockDim.x*gridDim.x;
    }
    cuseed[id%NRAND] = localState;
}

__device__ float3 setPositronRange(float3 xtemp, float4 vxtemp, curandState* plocalState, int usedirection)
/*******************************************************************
c*    Finds positron range according to its energy                 *
c*                                                                 *
c*    Input:                                                       *
c*    xtemp -> current position, vxtemp -> vx vy vz energy,        *
c*    usedirection -> 0 sample positron direction or 1 predefined  *
c*    Output:                                                      *
c*      position + range                                           *
c******************************************************************/
{
    int id = blockDim.x*blockIdx.x+threadIdx.x;
    float3 distance;
    float ekin = vxtemp.w/1e6; //transfer to MeV
    //float Zeff = 7.2222; //1/9*1+8/9*8;
    //float Aeff = 13; //Zeff/(1/9*1/1+8/9*8/16)
    float b1 = 5.44040782;//4.569*Aeff/powf(Zeff,1.209);
    float b2 = 0.369516529; // 1/(2.873-0.02309*Zeff);
    float Rex = 0.1*b1*ekin*ekin/(b2+ekin);
    float sigma = Rex/(2*1.0f); //put material density here
    distance.x = sigma*curand_normal(plocalState);
    distance.y = sigma*curand_normal(plocalState);
    distance.z = sigma*curand_normal(plocalState);
    float r=sqrtf(distance.x*distance.x+distance.y*distance.y+distance.z*distance.z);
    if(usedirection)
    {
        float tmp = sqrtf(vxtemp.x*vxtemp.x+vxtemp.y*vxtemp.y+vxtemp.z*vxtemp.z);
        distance.x = r*vxtemp.x/tmp;//reassign direction
        distance.y = r*vxtemp.y/tmp;
        distance.z = r*vxtemp.z/tmp;
    }

    float s = 0, step =100;
    int4 absvoxtemp = getAbsVox(xtemp);
    while(s<r)
    {
        step = 1000;
        if(absvoxtemp.w>0)
        {           
            b1 = (Offsetx_gBrachy+(absvoxtemp.x+(distance.x>0))*dx_gBrachy-xtemp.x)/distance.x;//remaining voxel length along x direction
            if(step > b1) {step = b1;absvoxtemp.w=1;}
            b1 = (Offsety_gBrachy+(absvoxtemp.y+(distance.y>0))*dy_gBrachy-xtemp.y)/distance.y;
            if(step > b1) {step = b1;absvoxtemp.w=2;}
            b1 = (Offsetz_gBrachy+(absvoxtemp.z+(distance.z>0))*dz_gBrachy-xtemp.z)/distance.z;
            if(step > b1) {step = b1;absvoxtemp.w=3;}

            if(absvoxtemp.w == 1) absvoxtemp.x += (distance.x>0)?1:-1;
            else if(absvoxtemp.w == 2) absvoxtemp.y += (distance.y>0)?1:-1;
            else absvoxtemp.z += (distance.z>0)?1:-1;
            b2 = tex3D(dens_tex, absvoxtemp.x, absvoxtemp.y, absvoxtemp.z);
            step = step*r;
            s += step*b2;

            if(s > r) step += (r-s)/b2;

        }
        else
        {
            step += (r-s)/0.0012905;
            s = r + 100;//make s > r
        }

        xtemp.x += step*distance.x/r;
        xtemp.y += step*distance.y/r;
        xtemp.z += step*distance.z/r; 

        if(xtemp.x < Offsetx_gBrachy || xtemp.x > (Offsetx_gBrachy+Unxvox*dx_gBrachy)) absvoxtemp.w = -1;
        if(xtemp.y < Offsety_gBrachy || xtemp.y > (Offsety_gBrachy+Unyvox*dy_gBrachy)) absvoxtemp.w = -1;
        if(xtemp.z < Offsetz_gBrachy || xtemp.z > (Offsetz_gBrachy+Unzvox*dz_gBrachy)) absvoxtemp.w = -1;     
    }
    return xtemp;
}

__device__ float sampleEkPositron(int type, float* d_coef,curandState* plocalState)
/*******************************************************************
c*    Finds positron energy according to fitted parameters         *
c*                                                                 *
c*    Input:                                                       *
c*    type -> source type, d_coef -> fitted coefficients           *
c*    Output:                                                      *
c*      kinetic energy in eV                                       *
c*    Comments:                                                    *
c*      coefficients can be fitted from calculated spectrum        *
c*      refer to Levin et al PMB 44, 781-799                       *
c******************************************************************/
{
    float u=100, E=0, sumE=0;
    while(u> sumE)
    {
        E=curand_uniform(plocalState)*(d_coef[8*type]-0.511)+0.511;
        u=d_coef[8*type+1]*curand_uniform(plocalState);
        sumE=0;
        for(int i=0;i<6;i++)
            sumE+=d_coef[8*type+2+i]*powf(E,5-i);
    }
    return (E-0.511)*1e6;//consistent in unit eV
}

__device__ float3 getPositionFromShape(int index, int shapeindex, float* shapecoeff, curandState* plocalState)
/*******************************************************************
c*    Sample position uniformly inside a volume                    *
c*                                                                 *
c*    Input:                                                       *
c*    type -> source type, shapeindex -> shape type                *
c*    Output:                                                      *
c*      position                                                   *
c******************************************************************/
{
    float3 position;
    //int id = blockDim.x*blockIdx.x+threadIdx.x;
    if(shapeindex<0 || shapeindex>2) shapeindex=0;
    if(shapeindex==0)//box
    {
        position.x=shapecoeff[6*index]+shapecoeff[6*index+3]*(-1+2*curand_uniform(plocalState))*0.5;
        position.y=shapecoeff[6*index+1]+shapecoeff[6*index+4]*(-1+2*curand_uniform(plocalState))*0.5;
        position.z=shapecoeff[6*index+2]+shapecoeff[6*index+5]*(-1+2*curand_uniform(plocalState))*0.5;
    }
    else if(shapeindex==1)//cylinder
    {
        float phi=TWOPI*curand_uniform(plocalState);
        float r= shapecoeff[6*index+3]*sqrtf(curand_uniform(plocalState));
        position.x=shapecoeff[6*index]+r*cosf(phi);
        position.y=shapecoeff[6*index+1]+r*sinf(phi);
        position.z=shapecoeff[6*index+2]+shapecoeff[6*index+4]*(-1+2*curand_uniform(plocalState))*0.5;
    }
    else if(shapeindex==2)//sphere
    {
        float phi=TWOPI*curand_uniform(plocalState);
        float costheta=-1+2*curand_uniform(plocalState);
        float r= shapecoeff[6*index+3]*cbrtf(curand_uniform(plocalState));
        position.x=shapecoeff[6*index]+r*sqrtf(1-costheta*costheta)*cosf(phi);
        position.y=shapecoeff[6*index+1]+r*sqrtf(1-costheta*costheta)*sinf(phi);
        position.z=shapecoeff[6*index+2]+r*costheta;
    }
    return position;
}
__global__ void setPosition(int nsource, unsigned int totalatom, float tref, float t, unsigned int* d_natom, unsigned int* d_sumpartial, int* d_type, int* d_shape, 
    float* d_shapecoeff, float* d_halftime, float* d_decayRatio, float* d_coef, int useprange)
/*******************************************************************
c*    Sample photon information from a positron source             *
c*    for a given time period                                      *
c******************************************************************/
{
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    curandState localState = cuseed[id%NRAND];
    int sourceindex=0, istart, iend, imid;
    float randno, phi;
    double ptime=0, tmp=0;
    float3 xtemp;
    float4  vxtemp;
    while(id<totalatom)
    {
#if BISEARCH == 0
        for(sourceindex=0;sourceindex<nsource;sourceindex++)
        {
            if(id<d_sumpartial[sourceindex]) break;
        }
#else
        if(id < d_sumpartial[0]) sourceindex = 0;
        else
        {
            istart = 0;
            iend = nsource-1;
            while(iend - istart >1)
            {
                imid = (iend+istart)/2;
                if(id<d_sumpartial[imid]) iend = imid;
                else istart = imid;
            }
            sourceindex = iend;
        }
#endif       
        tmp = double(-d_halftime[d_type[sourceindex]]*1.442695);
        ptime = tmp*log(curand_uniform_double(&localState));      
        if(ptime<t) 
        {
            atomicSub(&(d_natom[sourceindex]),1);
            randno=curand_uniform(&localState);
            if(randno < d_decayRatio[d_type[sourceindex]]) 
            {
                int ind = atomicAdd(&d_curemitted,1);
                xtemp = getPositionFromShape(sourceindex,d_shape[sourceindex],d_shapecoeff, &localState);

                if(useprange == 1)
                {
                    vxtemp.w = sampleEkPositron(d_type[sourceindex], d_coef, &localState);
                    xtemp = setPositronRange(xtemp, vxtemp,&localState,0);
                }

                randno=-1+2*curand_uniform(&localState);
                phi = TWOPI*curand_uniform(&localState);
                vxtemp.x = sqrtf(1-randno*randno)*cosf(phi);
                vxtemp.y = sqrtf(1-randno*randno)*sinf(phi);
                vxtemp.z = randno;
                //printf("%d vx vy vz are %f %f %f\n", id, vxtemp.x,vxtemp.y,vxtemp.z);
                x_gBrachy[(2*ind)%NPART]=xtemp; //avoid boundary excess
                x_gBrachy[(2*ind+1)%NPART]=xtemp;
                d_time[(2*ind)%NPART]=double(tref+ptime)*1e6;
                d_time[(2*ind+1)%NPART]=double(tref+ptime)*1e6;
                d_eventid[(2*ind)%NPART]=ind;
                d_eventid[(2*ind+1)%NPART]=ind;
                
                phi = TWOPI*curand_uniform(&localState);
                randno=curand_normal(&localState)*NonAngle_d;
                vxtemp.w=MC2+randno*MC2/2.0;//Noncollinearity
                vx_gBrachy[(2*ind)%NPART]=vxtemp;
                rotate(&(vxtemp.x), &(vxtemp.y), &(vxtemp.z),-cosf(randno), phi);
                vxtemp.w=MC2-randno*MC2/2.0;
                vx_gBrachy[(2*ind+1)%NPART]=vxtemp;              
            }
        }
        id+=blockDim.x*gridDim.x;
    }
    cuseed[id%NRAND]=localState;   
}

__global__ void setPositionForPhoton(int total,int curpar, int useprange)
/*******************************************************************
c*    Sample photon information from positron PSF                  *
c******************************************************************/
{
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    curandState localState = cuseed[id%NRAND];
    
    float randno, phi;
    float3 xtemp, tmptmp;
    float4  vxtemp;
    while(id<total)
    {
        xtemp = x_gBrachy[id];
        vxtemp = vx_gBrachy[id];
        if(useprange == 1)
        {
            xtemp = setPositronRange(xtemp, vxtemp, &localState, 1);
        }
        randno = -1+2*curand_uniform(&localState);
        phi = TWOPI*curand_uniform(&localState);
        vxtemp.x = sqrtf(1-randno*randno)*cosf(phi);
        vxtemp.y = sqrtf(1-randno*randno)*sinf(phi);
        vxtemp.z = randno;

        x_gBrachy[id]=xtemp; //avoid boundary excess
        x_gBrachy[(id+total)%NPART]=xtemp;
        d_eventid[id]=curpar+id;
        d_eventid[(id+total)%NPART]=curpar+id;
        
        phi = TWOPI*curand_uniform(&localState);
        randno=curand_normal(&localState)*NonAngle_d;
        vxtemp.w=MC2+randno*MC2/2.0;
        vx_gBrachy[id]=vxtemp;
        rotate(&(vxtemp.x), &(vxtemp.y), &(vxtemp.z),-cosf(randno), phi);
        vxtemp.w=MC2-randno*MC2/2.0;
        vx_gBrachy[(id+total)%NPART]=vxtemp;

        id+=blockDim.x*gridDim.x;
    }
    cuseed[id%NRAND]=localState;   
}

//digitizer
void __global__ setSitenum(int total, Event* events_d,int depth)
/*******************************************************************
c*    set site index according to depth                            *
c******************************************************************/
{
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    while(id<total)
    {
        switch(depth)
        {
            case 0:
            {                
                events_d[id].siten=0;
                break;
            }
            case 1:
            {
                events_d[id].siten=events_d[id].pann;
                break;
            }
            case 2:
            {
                events_d[id].siten=events_d[id].pann*moduleN+events_d[id].modn;
                break;
            }
            case 3:
            {
                events_d[id].siten=events_d[id].pann*moduleN*crystalN+events_d[id].modn*crystalN+events_d[id].cryn;
                break;
            }
        }
        id+=blockDim.x*gridDim.x;
    }
}
void __global__ energywindow(int* counts, Event* events,int total, float thresholder, float upholder)
//this is for the energy window part in digitizer
{
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    int num=0;
    while(id<total)
    {
        if(events[id].E<thresholder || events[id].E>upholder)
        {
            events[id].t=MAXT;
            num++;
        }
        id+=blockDim.x*gridDim.x;
    }
    if(num) atomicSub(counts,num);
}
void __global__ deadtime(int* counts,Event* events,int total, float interval, int deadtype)
{
    //this is the deadtime part in digitizer
   //deadtype 0 for paralysable, 1 for non
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    int start,current,i,k;
    float tdead;
    while(id<total)
    {
        start=id;
        if(start==0||events[start].siten!=events[start-1].siten||events[start].t>(events[start-1].t+interval))//find the start index
        {
            current=start;
            i=current+1;
            k=0;
            tdead=events[start].t;
            while(i<total)
            {
                while(events[i].siten==events[current].siten && events[i].t<(tdead+interval))
                {
                    //events[current].E+=events[i].E;
                    if(!deadtype) {
                        tdead=events[i].t;    //paralyzable accounts for pile-up effect
                        //events[current].t=events[i].t;
                    }
                    events[i].t=MAXT;
                    i++;
                    k++;
                    if(i==total) break;
                }
                if(i==total) break;
                if(events[i].siten!=events[i-1].siten||events[i].t>(events[i-1].t+interval))
                    break;
                current=i;
                tdead=events[current].t;
                i++;
            }
            atomicSub(counts,k);
        }
        id+=blockDim.x*gridDim.x;
    }
}
void __global__ addnoise(int* counts, Event* events_d, float lambda, float Emean, float sigma, float interval)
{
    //this is the noise part for digitizer
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    Event events[6];
    float t=id*interval;//0;
    curandState localstate = cuseed[id%NRAND];
    int i=0, ind=0;
    while(t<(id+1)*interval)
    {
        t+=-__logf(curand_uniform(&localstate))*lambda;
        if(t<(id+1)*interval)
        {
            events[i].t=t;//+id*interval;
            events[i].E=Emean+sigma*curand_normal(&localstate);//2.355;
            events[i].x=curand_uniform(&localstate);//need to be implemented to be matched to global coordinates
            events[i].y=curand_uniform(&localstate);
            events[i].z=curand_uniform(&localstate);
            events[i].parn=-1;
            events[i].pann=int(dev_totalPanels*curand_uniform(&localstate));
            events[i].modn=int(moduleN*curand_uniform(&localstate));
            events[i].cryn=int(crystalN*curand_uniform(&localstate));
            events[i].siten=events[i].pann*moduleN*crystalN+events[i].modn*crystalN+events[i].cryn;
            i=(i+1)%6;
            if(!i)
            {
                ind=atomicAdd(counts,6);
                for(int j=0; j<6; j++)
                    events_d[ind+j]=events[j];
            }
        }
    }
    cuseed[id]=localstate;
    ind=atomicAdd(counts,i);
    for(int j=0; j<i; j++)
        events_d[ind+j]=events[j];
}

int __device__ adder(int* counts_d, Event* events_d, Event event)
{
    //this is the adder part in digitizer
    for(int i=0; i < counts_d[0]; i++)
    {
        if(event.siten == events_d[i].siten)
        {
            events_d[i].x = (events_d[i].x*events_d[i].E + event.x*event.E)/(events_d[i].E + event.E);
            events_d[i].y = (events_d[i].y*events_d[i].E + event.y*event.E)/(events_d[i].E + event.E);
            events_d[i].z = (events_d[i].z*events_d[i].E + event.z*event.E)/(events_d[i].E + event.E);
            events_d[i].E = (events_d[i].E + event.E);
            return 1;
        }
    }
    //no recorded event inside the the same crystal
    events_d[counts_d[0]]=event;
    counts_d[0]++;
    return 1;
}
int __device__ readout(int* counts_d, Event* events_d,int depth, int policy)
{
    //this is for the readout part in digitizer
    //depth means the readout level. 0,1,2,3 represents world,panel,module,cry
    //policy 0,1 for winnertakeall and energy centroid   
    if(depth==3) return 1;
    if(policy==1) depth = 2;
    //the readout part
    switch(depth)
    {
        case 0:
        {
            for(int i=0; i<counts_d[0]; i++)
                events_d[i].siten=0;
            break;
        }
        case 1:
        {
            for(int i=0; i<counts_d[0]; i++)
                events_d[i].siten=events_d[i].pann;
            break;
        }
        case 2:
        {
            for(int i=0; i<counts_d[0]; i++)
                events_d[i].siten=events_d[i].pann*moduleN+events_d[i].modn;
            break;
        }
    }
    int ind=0;
    for(int i=0; i<counts_d[0]; i++)
    {
        Event event0 = events_d[i];
        if(event0.t>MAXT*0.1) continue;        
        for(int j=i+1; j<counts_d[0]; j++)
        {
            Event event = events_d[j];
            if((event.parn==event0.parn)&&(event.siten == event0.siten))
            {
                if(policy==1)
                {
                    event0.x = (event0.x*event0.E + event.x*event.E)/(event0.E + event.E);
                    event0.y = (event0.y*event0.E + event.y*event.E)/(event0.E + event.E);
                    event0.z = (event0.z*event0.E + event.z*event.E)/(event0.E + event.E);
                    event0.E = (event0.E + event.E);
                    events_d[j].t=MAXT;
                    continue;
                }
                event0=(event0.E>event.E)?event0:event;
                events_d[j].t=MAXT;
            }
        }
        events_d[ind]=event0;
        ind++;
    }
    counts_d[1]=ind;
    return 1;
}
void __global__ blur(int total, Event* events, int Eblurpolicy, float Eref, float Rref, float slope, float Spaceblur)
{
    //this is the energy blurring part in digitizer
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float R=0;
    curandState localstate=cuseed[i%NRAND];
    while(i<total)
    {
        if(Eblurpolicy==0) R = sqrt(Eref/events[i].E)*Rref;
        if(Eblurpolicy==1) R = Rref+slope*(events[i].E-Eref)/1e6; //slope in 1/MeV
        if(R<0) R=0;
        events[i].E += curand_normal(&localstate)*R*events[i].E/2.35482;
//other distribution of energy blurring need to be implemented
        if(Spaceblur>0)
        {   
            events[i].x+=Spaceblur*curand_normal(&localstate);
            events[i].y+=Spaceblur*curand_normal(&localstate);
            events[i].z+=Spaceblur*curand_normal(&localstate);
        }

        i+=blockDim.x*gridDim.x;
    }
    cuseed[i%NRAND]=localstate;
}

__global__ void photonde(Event* events_d, int* counts_d, int nactive, int bufferID, float* dens, int *mat, int *panelID, float *lenx, float *leny, float *lenz,
                       float *MODx, float *MODy, float *MODz, float *Msx, float *Msy, float *Msz, float *LSOx, float *LSOy, float *LSOz, float *sx, float *sy, float *sz,
                       float *ox, float *oy, float *oz, float *dx, float *dy, float *dz, float *UXx, float *UXy, float *UXz,
                       float *UYx, float *UYy, float *UYz,float *UZx, float *UZy, float *UZz)
/*******************************************************************
c*    Transports a photon until it either escapes from the         *
c*    detector or its energy drops below EabsPhoton                *
c*                                                                 *
c******************************************************************/
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int tid=threadIdx.x;

    Event events[4];
    int counts[2]= {0,4};
    Event event;

    float tempDen=0.0f;
    int tempMat=0;

    __shared__ int nsstktemp;
    if(tid==0)
    {
        nsstktemp = 0;
    }
    __syncthreads();
    __shared__ float sftemp[NSSTACKSHARED];
    __shared__ int sidtemp[NSSTACKSHARED];
    extern __shared__ float s[];
    float *dens_S = s;
    int *mat_S = (int*)&dens_S[2];
    int *panelID_S = (int*)&mat_S[2];
    float *lenx_S = (float*)&panelID_S[dev_totalPanels];
    float *leny_S = (float*)&lenx_S[dev_totalPanels];
    float *lenz_S = (float*)&leny_S[dev_totalPanels];
    float *MODx_S = (float*)&lenz_S[dev_totalPanels];
    float *MODy_S = (float*)&MODx_S[dev_totalPanels];
    float *MODz_S = (float*)&MODy_S[dev_totalPanels];
    float *Msx_S = (float*)&MODz_S[dev_totalPanels];
    float *Msy_S = (float*)&Msx_S[dev_totalPanels];
    float *Msz_S = (float*)&Msy_S[dev_totalPanels];
    float *LSOx_S = (float*)&Msz_S[dev_totalPanels];
    float *LSOy_S = (float*)&LSOx_S[dev_totalPanels];
    float *LSOz_S = (float*)&LSOy_S[dev_totalPanels];
    float *sx_S = (float*)&LSOz_S[dev_totalPanels];
    float *sy_S = (float*)&sx_S[dev_totalPanels];
    float *sz_S = (float*)&sy_S[dev_totalPanels];
    float *ox_S = (float*)&sz_S[dev_totalPanels];
    float *oy_S = (float*)&ox_S[dev_totalPanels];
    float *oz_S = (float*)&oy_S[dev_totalPanels];
    float *dx_S = (float*)&oz_S[dev_totalPanels];
    float *dy_S = (float*)&dx_S[dev_totalPanels];
    float *dz_S = (float*)&dy_S[dev_totalPanels];
    float *UXx_S = (float*)&dz_S[dev_totalPanels];
    float *UXy_S = (float*)&UXx_S[dev_totalPanels];
    float *UXz_S = (float*)&UXy_S[dev_totalPanels];
    float *UYx_S = (float*)&UXz_S[dev_totalPanels];
    float *UYy_S = (float*)&UYx_S[dev_totalPanels];
    float *UYz_S = (float*)&UYy_S[dev_totalPanels];
    float *UZx_S = (float*)&UYz_S[dev_totalPanels];
    float *UZy_S = (float*)&UZx_S[dev_totalPanels];
    float *UZz_S = (float*)&UZy_S[dev_totalPanels];

    if(tid==0)
    {
        for (int i=0; i<2; i++)
        {
            mat_S[i]=mat[i];
            dens_S[i]=dens[i];
        }
        for(int i=0; i<dev_totalPanels; i++)
        {
            panelID_S[i]=panelID[i];
            lenx_S[i]=lenx[i];
            leny_S[i]=leny[i];
            lenz_S[i]=lenz[i];
            MODx_S[i]=MODx[i];
            MODy_S[i]=MODy[i];
            MODz_S[i]=MODz[i];
            Msx_S[i]=Msx[i];
            Msy_S[i]=Msy[i];
            Msz_S[i]=Msz[i];
            LSOx_S[i]=LSOx[i];
            LSOy_S[i]=LSOy[i];
            LSOz_S[i]=LSOz[i];
            sx_S[i]=sx[i];
            sy_S[i]=sy[i];
            sz_S[i]=sz[i];
            ox_S[i]=ox[i];
            oy_S[i]=oy[i];
            oz_S[i]=oz[i];
            dx_S[i]=dx[i];
            dy_S[i]=dy[i];
            dz_S[i]=dz[i];
            UXx_S[i]=UXx[i];
            UXy_S[i]=UXy[i];
            UXz_S[i]=UXz[i];
            UYx_S[i]=UYx[i];
            UYy_S[i]=UYy[i];
            UYz_S[i]=UYz[i];
            UZx_S[i]=UZx[i];
            UZy_S[i]=UZy[i];
            UZz_S[i]=UZz[i];
        }//*/
    }
    __syncthreads();


//  obtain current id on thread
    curandState localState = cuseed[id%NRAND];
    if( id < nactive )
    {
        if(d_time[id]>0)
        {
            float3 xtemp = x_gBrachy[id];
            float4 vxtemp = vx_gBrachy[id];
            double tof = d_time[id];
            int eid = d_eventid[id];
            
            // change global coordinates to local coordinates
            int paID=-1;
            float3 xtemp2;
            float4 vxtemp2;
            //get the panelid crystal id that the particle enters???
            for (int i=0; i<dev_totalPanels; i++)
            {
                //new coordinates 
                float tempx=(xtemp.x-ox_S[i])*UXx_S[i]+(xtemp.y-oy_S[i])*UXy_S[i]+(xtemp.z-oz_S[i])*UXz_S[i];
                float tempy=(xtemp.x-ox_S[i])*UYx_S[i]+(xtemp.y-oy_S[i])*UYy_S[i]+(xtemp.z-oz_S[i])*UYz_S[i];
                float tempz=(xtemp.x-ox_S[i])*UZx_S[i]+(xtemp.y-oy_S[i])*UZy_S[i]+(xtemp.z-oz_S[i])*UZz_S[i];

                float tempvx=vxtemp.x*UXx[i]+vxtemp.y*UXy[i]+vxtemp.z*UXz[i];//component along different directions
                float tempvy=vxtemp.x*UYx[i]+vxtemp.y*UYy[i]+vxtemp.z*UYz[i];
                float tempvz=vxtemp.x*UZx[i]+vxtemp.y*UZy[i]+vxtemp.z*UZz[i];

                float tempx2=0.0f;
                if(tempvx*dx_S[i]>=0)
                {

                    float tempy2=tempy-tempx/tempvx*tempvy;
                    float tempz2=tempz-tempx/tempvx*tempvz;

                    if(abs(tempy2)<leny_S[i]/2 & abs(tempz2)<lenz_S[i]/2)
                    {
                        xtemp2.x=tempx2;
                        xtemp2.y=tempy2;
                        xtemp2.z=tempz2;
                        //xtemp2.w=xtemp.w;
                        tof += -tempx/(29979.2458*tempvx);

                        vxtemp2.x=tempvx;
                        vxtemp2.y=tempvy;
                        vxtemp2.z=tempvz;
                        vxtemp2.w=vxtemp.w;
                        paID=panelID_S[i];
                        break;
                    }
                }
                float tempy2=tempy-tempx/tempvx*tempvy;
                float tempz2=tempz-tempx/tempvx*tempvz;
                xtemp2.x=tempx2;
                xtemp2.y=tempy2;
                xtemp2.z=tempz2;
                //xtemp2.w=xtemp.w;

                vxtemp2.x=tempvx;
                vxtemp2.y=tempvy;
                vxtemp2.z=tempvz;
                vxtemp2.w=vxtemp.w;
                paID=-1;
            }

    //      Loop until it either escapes or is absorbed:
            float lammin,s;
            for(;;)
            {
                if (paID==-1)
                    break;
                //      Get lambda from the minimum lambda at the current energy:
                lammin = lamwckde(vxtemp2.w); //0.1;//make sure use the one corresponding to detector
                s = -lammin*__logf(curand_uniform(&localState));

                //      Get the coordinates of the photon after passing a free length
                xtemp2.x += s*vxtemp2.x;
                xtemp2.y += s*vxtemp2.y;
                xtemp2.z += s*vxtemp2.z;
                tof += s/29979.2458;
                // if out of panel
                if (paID==-1|abs(xtemp2.y)>leny_S[paID]*0.5| abs(xtemp2.z)>lenz_S[paID]*0.5 |(xtemp2.x*dx_S[paID])<0 |(xtemp2.x*dx_S[paID])>lenx_S[paID])
                {
                    break;
                }

                int m_id=-1; //material id
                int M_id=-1; // module id
                int L_id=-1; // LSO id

                crystalSearch(xtemp2,leny_S[paID],lenz_S[paID],MODy_S[paID],MODz_S[paID],Msy_S[paID],Msz_S[paID], LSOy_S[paID],LSOz_S[paID],sy_S[paID],sz_S[paID],dy_S[paID], dz_S[paID], &m_id, &M_id, &L_id);

                tempDen = dens_S[m_id];
                tempMat = mat_S[m_id];

                //  Apply Woodcock trick:
                float lamden = lammin*tempDen;
                float prob = 1.0-lamden*itphip_G(tempMat, vxtemp2.w);
                if(prob<0) prob=0;
                float randno = curand_uniform(&localState);
                //  Compton:
                //  No real event; continue jumping:
                if (randno < prob)
                    continue;

                prob += lamden*icptip(tempMat, vxtemp2.w);
                if (randno < prob)
                {
                    float efrac, costhe;
                    comsam(vxtemp2.w, &localState, &efrac, &costhe);//, tempMat);
                    float de = vxtemp2.w * (1.0f-efrac);
                    float phi = TWOPI*curand_uniform(&localState);

                    //record events
                    if (nsstktemp!= NSSTACKSHARED && m_id==0)//only events inside crystal can be recorded
                    {
                        int ind = atomicAdd(&nsstktemp,5);
                        sidtemp[ind] = id+bufferID;
                        sidtemp[ind+1] = paID;
                        sidtemp[ind+2] = M_id;
                        sidtemp[ind+3] = L_id;
                        sidtemp[ind+4] = 1;

                        event.parn= id+bufferID;
                        event.cryn=L_id;
                        event.modn=M_id;
                        event.pann=paID;
                        event.siten=event.pann*moduleN*crystalN+event.modn*crystalN+event.cryn;//maybe should use event.pann-1 or event.cryn-1 to make sure siten start from 0
                        event.eventid = eid;

                        sftemp[ind] = de;//s;
                        sftemp[ind+1] = tof;//s;s;//
                        sftemp[ind+2] = xtemp2.x;
                        sftemp[ind+3] = xtemp2.y;
                        sftemp[ind+4] = xtemp2.z;
                        event.E = de;
                        event.t = tof;
                        event.x = xtemp2.x;
                        event.y = xtemp2.y;
                        event.z = xtemp2.z;
                        //ind=atomicAdd(counts_d,1);
                        //events_d[ind]=event;
                        adder(counts,events,event);
                    }

                    vxtemp2.w -= de;
                    if (vxtemp2.w < eabsph)
                    {

                        if (nsstktemp!= NSSTACKSHARED && m_id==0)
                        {
                            int ind = atomicAdd(&nsstktemp,5);
                            sidtemp[ind] = id+bufferID;
                            sidtemp[ind+1] = paID;
                            sidtemp[ind+2] = M_id;
                            sidtemp[ind+3] = L_id;
                            sidtemp[ind+4] = 2;

                            event.parn=id+bufferID;
                            event.cryn=L_id;
                            event.modn= M_id;
                            event.pann=paID;
                            event.siten=event.pann*moduleN*crystalN+event.modn*crystalN+event.cryn;
                            event.eventid = eid;

                            sftemp[ind] = vxtemp2.w;//s;
                            sftemp[ind+1] = tof;//s;s;//
                            sftemp[ind+2] = xtemp2.x;
                            sftemp[ind+3] = xtemp2.y;
                            sftemp[ind+4] = xtemp2.z;

                            event.E = vxtemp2.w;
                            event.t = tof;
                            event.x =xtemp2.x;
                            event.y = xtemp2.y;
                            event.z = xtemp2.z;
                            //ind=atomicAdd(counts_d,1);
                            //events_d[ind]=event;
                            adder(counts,events,event);
                        }
                        break;
                    }

                    rotate(&vxtemp2.x,&vxtemp2.y,&vxtemp2.z,costhe,phi);
                    continue;
                }

    //  Rayleigh:
                prob += lamden*irylip(tempMat, vxtemp2.w);
                if (randno < prob)
                {
                    float costhe;
                    rylsam(vxtemp2.w, tempMat, &localState, &costhe);
                    float phi = TWOPI*curand_uniform(&localState);
#if RECORDRayleigh==1
                    if (nsstktemp!= NSSTACKSHARED && m_id==0)
                    {
                        int ind = atomicAdd(&nsstktemp,5);
                        sidtemp[ind] = id+bufferID;
                        sidtemp[ind+1] = paID;
                        sidtemp[ind+2] = M_id;
                        sidtemp[ind+3] = L_id;
                        sidtemp[ind+4] = 3;
                        sftemp[ind] = 0.0f;//s;
                        sftemp[ind+1] = tof;//s;s;//
                        sftemp[ind+2] = xtemp2.x;
                        sftemp[ind+3] = xtemp2.y;
                        sftemp[ind+4] = xtemp2.z;
                    }
#endif
                    rotate(&vxtemp2.x,&vxtemp2.y,&vxtemp2.z,costhe,phi);
                    continue;
                }
    //  Photoelectric:
                //if(id <1) printf("photo den lamda prob are %f %f %f\n", tempDen, lamden,1-prob);

                if (nsstktemp!= NSSTACKSHARED && m_id==0)
                {
                    int ind = atomicAdd(&nsstktemp,5);
                    sidtemp[ind] = id+bufferID;
                    sidtemp[ind+1] = paID;
                    sidtemp[ind+2] = M_id;
                    sidtemp[ind+3] = L_id;
                    sidtemp[ind+4] = 4;

                    sftemp[ind] = vxtemp2.w;
                    sftemp[ind+1] = tof;//s;
                    sftemp[ind+2] = xtemp2.x;
                    sftemp[ind+3] = xtemp2.y;
                    sftemp[ind+4] = xtemp2.z;

                    event.parn=id+bufferID;
                    event.cryn=L_id;
                    event.modn= M_id;
                    event.pann=paID;
                    event.siten=event.pann*moduleN*crystalN+event.modn*crystalN+event.cryn;
                    event.eventid = eid;
                    event.E = vxtemp2.w;
                    event.t = tof;
                    event.x = xtemp2.x;
                    event.y = xtemp2.y;
                    event.z = xtemp2.z;
                    //ind=atomicAdd(counts_d,1);
                    //events_d[ind]=event;
                    adder(counts, events, event);//this is for digitizer
                }
                break;
            }
            if(counts[0])
            {
                readout(counts,events,rdepth_d,rpolicy_d);
                int ind=atomicAdd(counts_d,counts[1]);
                for(int i=0; i<counts[1]; i++)
                    events_d[ind+i]=events[i];
            }//*/
        }
        //id+=blockDim.x*gridDim.x;
    }
    __syncthreads();
    if(id<NRAND) cuseed[id] = localState;
    __shared__ int istart;
    if(threadIdx.x==0)
    {
        //printf("nsstktemp1 = %d\n",nsstktemp);
        istart = atomicAdd(&nsstk, nsstktemp);
        //printf("istart = %d\n",istart);
    }
      __syncthreads();

   // if(id==0) printf("total events=%d\ncurrent total hits=%d\n", counts_d[0],nsstk);
  
    for(int i = 0; i < 1+(nsstktemp)/blockDim.x; i++)
    {
        if(nsstktemp == 0)
            break;

        int ind = istart + i*blockDim.x + tid;


        if(ind < istart + nsstktemp && ind<NSSTACK)
        {
            sf[ind] = sftemp[i*blockDim.x + tid];//this is for hits events
            sid[ind] = sidtemp[i*blockDim.x + tid];
        }
    }
    __syncthreads();//*/

}

// crystal index and material type
__device__ void crystalSearch(float3 xtemp2,float leny_S,float lenz_S,float MODy_S,float MODz_S,float Msy_S,float Msz_S,float LSOy_S,float LSOz_S,float sy_S,float sz_S,float dy_S, float dz_S, int *m_id, int *M_id, int *L_id)
{
    float x=xtemp2.x;
    float y=xtemp2.y;
    float z=xtemp2.z;
    for(int tmp=0;tmp<Nsurface_d;tmp++)
    {
        if((surface_d[tmp*10+0]*x*x+surface_d[tmp*10+1]*y*y+surface_d[tmp*10+2]*z*z+surface_d[tmp*10+3]*x*y+surface_d[tmp*10+4]*x*z+surface_d[tmp*10+5]*y*z
        +surface_d[tmp*10+6]*x+surface_d[tmp*10+7]*y+surface_d[tmp*10+8]*z+surface_d[tmp*10+9])<0)
        {
            *m_id=1;
            return;
        }
    }
    

    y=leny_S/2+xtemp2.y;//*dy_S;
    z=lenz_S/2+xtemp2.z;//*dz_S;
    int M_id_y=floorf(y/(MODy_S+Msy_S))>0?int(y/(MODy_S+Msy_S)):0;
    int M_id_z=floorf(z/(MODz_S+Msz_S))>0?int(z/(MODz_S+Msz_S)):0;
    *M_id=M_id_z*moduleNy+M_id_y;

    y=y-M_id_y*(MODy_S+Msy_S);
    z=z-M_id_z*(MODz_S+Msz_S);
    if(y> MODy_S || z> MODz_S)
    {
        *m_id=1;
        return;
    }

    int L_id_y=floorf(y/(LSOy_S+sy_S))>0?int(y/(LSOy_S+sy_S)):0;
    int L_id_z=floorf(z/(LSOz_S+sz_S))>0?int(z/(LSOz_S+sz_S)):0;
    *L_id=L_id_z*crystalNy+L_id_y;

    y=y-L_id_y*(LSOy_S+sy_S);
    z=z-L_id_z*(LSOz_S+sz_S);
    if(y>LSOy_S || z> LSOz_S)
    {
        *m_id=1;
        return;
    }
    *m_id=0;//*/

}

#endif
