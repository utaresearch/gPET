#ifndef __GPET_CU__
#define __GPET_CU__
#include "gPET.h"
#include "externCUDA.h"

float3 xbufferRepeat[NPART];
float4 vxbufferRepeat[NPART];
double h_time[NPART];
int eventidbuffer[NPART];

using namespace std;

void simulateParticle(Particle particle,int ptype_h, int total_Panels)
{
    time_t start_time, end_time;
    float time_diff;
    start_time = clock();

    double* a= new double[NPART];//used for initialization
    memset(a,0,sizeof(double)*NPART);
    int naverage = 3;//assuming 3 singles will be recorded on average per photon
    //should change accordingly if long crystals are used
    int zero=0, temptemp[2]= {0,NPART*naverage};;

    Event* events_d;//used for recording singles
    cudaMalloc(&events_d,sizeof(Event)*(NPART*naverage));
    int* counts_d;
    cudaMalloc(&counts_d,sizeof(int)*naverage);

    int first=0, last = particle.NParticle, nactive_h=0, curparticle=0, nsstk_h=0, npar=0;
    size_t nShared = (total_Panels+2)*sizeof(int)+(30*total_Panels+2)*sizeof(float);

    if(ptype_h==0) npar=floor(NPART/2); // leaving space for photons --> doubling the number of positrons 
    else npar=NPART;

//  loop until all particles from ps file are done
    while(curparticle<particle.NParticle)
    {
        first=curparticle;
        last = first + npar -1;
        if(last>particle.NParticle-1)
            last=particle.NParticle-1;
                        
        nactive_h = last - first + 1; // particle number in a batch

//copy data, CUDA_CALL is defined in gPET.h 
        CUDA_CALL(cudaMemcpyToSymbol(d_time,a,sizeof(double)*NPART,0,cudaMemcpyHostToDevice));
        
        CUDA_CALL(cudaMemcpyToSymbol(x_gBrachy, &(particle.xbuffer[first]),sizeof(float3)*nactive_h, 0, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(vx_gBrachy, &(particle.vxbuffer[first]),sizeof(float4)*nactive_h, 0, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(d_eventid, &(particle.eventid[first]),sizeof(int)*nactive_h, 0, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(d_time, &(particle.time[first]),sizeof(double)*nactive_h, 0, cudaMemcpyHostToDevice));               

//      simulate a batch of particles
        if (ptype_h == 0 && nactive_h>0)
        {
            // put position for photon based on the PSF of positron
            setPositionForPhoton<<<NRAND/NTHREAD_PER_BLOCK_GPET, NTHREAD_PER_BLOCK_GPET>>>(nactive_h, curparticle, useprange_h);
            cudaDeviceSynchronize();
            nactive_h*=2;      
        }
//output data
#if OUTPUTPSF == 1
        CUDA_CALL(cudaMemcpyFromSymbol( &(xbufferRepeat[0]), x_gBrachy,sizeof(float3)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(vxbufferRepeat[0]), vx_gBrachy, sizeof(float4)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(eventidbuffer[0]), d_eventid, sizeof(int)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(h_time[0]), d_time, sizeof(double)*NPART, 0, cudaMemcpyDeviceToHost));
        //printf("copy data finished\n");
        ofstream out("./output/outsource.dat",ios::app|ios::binary);
        ofstream outid("./output/idsource.dat",ios::app|ios::binary);
        ofstream outt("./output/timesource.dat",ios::app|ios::binary);
        for(int ii=0;ii<NPART;ii++)
        {
            if(h_time[ii]>0)
            {
                out.write((char*) &(xbufferRepeat[ii]),sizeof(float3));
                out.write((char*) &(vxbufferRepeat[ii]),sizeof(float4));
                outid.write((char*) &(eventidbuffer[ii]),sizeof(int));
                outt.write((char*) &(h_time[ii]),sizeof(double));
            }
            
        }//*/
        out.close();
        outid.close();
        outt.close();
#endif
//simulate transport of photons in phantom
        photon<<<NRAND/NTHREAD_PER_BLOCK_GPET, NTHREAD_PER_BLOCK_GPET>>>(NPART);
        cudaDeviceSynchronize();
//output data
#if OUTPUTPSF == 2
        CUDA_CALL(cudaMemcpyFromSymbol( &(xbufferRepeat[0]), x_gBrachy,sizeof(float3)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(vxbufferRepeat[0]), vx_gBrachy, sizeof(float4)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(eventidbuffer[0]), d_eventid, sizeof(int)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(h_time[0]), d_time, sizeof(double)*NPART, 0, cudaMemcpyDeviceToHost));
        //printf("copy data finished\n");
        ofstream out("./output/outphantom.dat",ios::app|ios::binary);
        ofstream outid("./output/idphantom.dat",ios::app|ios::binary);
        ofstream outt("./output/timephantom.dat",ios::app|ios::binary);
        for(int ii=0;ii<NPART;ii++)
        {
            if(h_time[ii]>0)
            {
                out.write((char*) &(xbufferRepeat[ii]),sizeof(float3));
                out.write((char*) &(vxbufferRepeat[ii]),sizeof(float4));
                outid.write((char*) &(eventidbuffer[ii]),sizeof(int));
                outt.write((char*) &(h_time[ii]),sizeof(double));
            }
            
        }
        out.close();
        outid.close();
        outt.close();
#endif
//initialize
        CUDA_CALL(cudaMemcpyToSymbol(nsstk, &zero, sizeof(int), 0, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(counts_d,temptemp,sizeof(int)*2,cudaMemcpyHostToDevice));
//simulate transport of photons in detector       
        photonde<<<1+(NPART-1)/NTHREAD_PER_BLOCK_GPET, NTHREAD_PER_BLOCK_GPET,nShared>>>(events_d,counts_d,NPART, first,dens_panel, mat_panel, panelID, lengthx_panel, lengthy_panel, lengthz_panel,
                MODx_panel, MODy_panel, MODz_panel, Mspacex_panel, Mspacey_panel, Mspacez_panel,
                LSOx_panel, LSOy_panel, LSOz_panel, spacex_panel, spacey_panel, spacez_panel,
                offsetx_panel, offsety_panel, offsetz_panel, directionx_panel, directiony_panel, directionz_panel,
                UniXx_panel, UniXy_panel, UniXz_panel, UniYx_panel, UniYy_panel, UniYz_panel,
                UniZx_panel, UniZy_panel, UniZz_panel);
        cudaDeviceSynchronize();//*/   
        CUDA_CALL(cudaMemcpyFromSymbol(&nsstk_h, nsstk, sizeof(int), 0, cudaMemcpyDeviceToHost));
        printf("\nthere are %d Hits in this batch\n", nsstk_h/5);
//output data
#if OUTPUTHIT==1
        void *tempData;
        cudaMalloc( (void **) &tempData, nsstk_h*sizeof(int));
        CUDA_CALL(cudaMemcpyFromSymbol(tempData, sid, nsstk_h*sizeof(int), 0,cudaMemcpyDeviceToDevice));
        outputData(tempData, nsstk_h*sizeof(int), "./output/HitsID.dat", "ab");
        cudaFree(tempData);

        void *tempData2;
        CUDA_CALL(cudaMalloc( (void **) &tempData2, sizeof(float)*nsstk_h));
        CUDA_CALL(cudaMemcpyFromSymbol(tempData2, sf, sizeof(float)*nsstk_h, 0,cudaMemcpyDeviceToDevice));
        outputData(tempData2,sizeof(float)*(nsstk_h), "./output/Hits.dat", "ab");
        cudaFree(tempData2);
#endif
        int counts=0;
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after adder is "<<counts<<endl;
        //outevents(&counts,events_d,"adder.dat");

//insert proper digitizer module in the following part
//the module that can change the number of events should be followed by a sort
        //energy and spatial blur
        blur<<<NRAND/512,512>>>(counts,events_d,blurpolicy, Eref, Rref, Eslope, Sblur);
        cudaDeviceSynchronize();
        //outevents(&counts,events_d,"blur.dat");

        //energy window
        energywindow<<<NRAND/512,512>>>(counts_d,events_d, counts, Eth,2000000);
        cudaDeviceSynchronize();
        //quicksort_d(events_d,0,counts,3);// could try GPU sort for large NPART
        quicksort_h(events_d,0,counts,3);// if error occurs, use the CPU srt function
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after thresholder is "<<counts<<endl;
        //outevents(&counts,events_d,"thresholder1.dat");

        //deadtime part
        if(dlevel!=3)
        {
            setSitenum<<<counts/512+1,512>>>(counts,events_d,dlevel);
            cudaDeviceSynchronize();
            printf("set site number ok\n");
        } 
        orderevents(&counts,events_d);//make events globally ordered by site number, and then ordered by flight time in each volume
        deadtime<<<NRAND/512,512>>>(counts_d,events_d, counts, dtime, dtype);
        cudaDeviceSynchronize();
        cout<<"deadtime is ok\n";
        //quicksort_d(events_d,0,counts,3);
        quicksort_h(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after deadtime is "<<counts<<endl;
        //outevents(&counts,events_d,"./output/deadtime.dat");//*/

        energywindow<<<NRAND/512,512>>>(counts_d,events_d, counts, Ewinmin,Ewinmax);
        cudaDeviceSynchronize();
        //quicksort_d(events_d,0,counts,3);
        quicksort_h(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of singles is "<<counts<<endl;
        outevents(counts_d,events_d,"./output/singles.dat");//*/

        nactive_h = 0;
        curparticle = last+1;
    }
    cudaFree(events_d);
    cudaFree(counts_d);

    end_time = clock();
    time_diff = ((float)end_time - (float)start_time)/CLOCKS_PER_SEC;
    printf("\n\n****************************************\n");
    printf("Simulation time: %f s.\n\n",time_diff);
    printf("****************************************\n\n\n");
}

void sampleParticle(Source source, Isotopes isotopes, float tstart, float tend, int total_Panels)
{
    float ttemp, thalf;
    for(int i=0;i<source.NSource;i++)
    {
        thalf= isotopes.halftime[source.type[i]];
        source.natom[i]= floor(source.natom[i]*exp2(-tstart/thalf));
    }
    tend-=tstart; // set tstart as new reference time point

//initialize some parameters, avoid using gloabl variable by putting in the same function
    int *d_type, *d_shape;
    unsigned int* d_natom;
    CUDA_CALL(cudaMalloc((void **) &d_natom, sizeof(unsigned int)*source.NSource));
    CUDA_CALL(cudaMemcpy(d_natom,source.natom,sizeof(unsigned int)*source.NSource,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &d_type, sizeof(int)*source.NSource));
    CUDA_CALL(cudaMemcpy(d_type,source.type,sizeof(int)*source.NSource,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &d_shape, sizeof(int)*source.NSource));
    CUDA_CALL(cudaMemcpy(d_shape,source.shape,sizeof(int)*source.NSource,cudaMemcpyHostToDevice));
    float *d_shapecoeff;
    CUDA_CALL(cudaMalloc((void **) &d_shapecoeff, sizeof(float)*6*source.NSource));
    CUDA_CALL(cudaMemcpy(d_shapecoeff,source.shapecoeff,sizeof(float)*6*source.NSource,cudaMemcpyHostToDevice));
    
    float *d_halftime, *d_decayRatio, *d_coef;
    CUDA_CALL(cudaMalloc((void **) &d_halftime, sizeof(float)*isotopes.Ntype));
    CUDA_CALL(cudaMemcpy(d_halftime,isotopes.halftime,sizeof(float)*isotopes.Ntype,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &d_decayRatio, sizeof(float)*isotopes.Ntype));
    CUDA_CALL(cudaMemcpy(d_decayRatio,isotopes.decayRatio,sizeof(float)*isotopes.Ntype,cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMalloc((void **) &d_coef, sizeof(float)*isotopes.Ntype*8));
    CUDA_CALL(cudaMemcpy(d_coef,isotopes.coef,sizeof(float)*isotopes.Ntype*8,cudaMemcpyHostToDevice));

    unsigned int* sumpartial= new unsigned int[source.NSource], nemitted;
    unsigned int* d_sumpartial, totalatom;
    CUDA_CALL(cudaMalloc((void **) &d_sumpartial, sizeof(unsigned int)*source.NSource));
    
    printf("finish GPU memory transfer for source information\n");
    
    int enough=1;
    unsigned int preemitted = 0, curemitted=0;
    CUDA_CALL(cudaMemcpyToSymbol(d_curemitted, &curemitted,sizeof(unsigned int), 0, cudaMemcpyHostToDevice));

    double* a= new double[NPART];
    memset(a,0,sizeof(double)*NPART);

    time_t start_time, end_time;
    float time_diff;
    start_time = clock();

    int naverage = 3;//assuming 3 singles will be recorded on average per photon
    //should change accordingly if long crystals are used
    Event* events_d;
    CUDA_CALL(cudaMalloc(&events_d,sizeof(Event)*(NPART*naverage)));
    int* counts_d;
    CUDA_CALL(cudaMalloc(&counts_d,sizeof(int)*naverage));

    int temptemp[2]= {0,NPART*naverage};
    int zero=0, nsstk_h=0;
    size_t nShared = (total_Panels+2)*sizeof(int)+(30*total_Panels+2)*sizeof(float);

    for(int epoch=0;;epoch++)
    {
        enough=1;
        
        thalf= isotopes.halftime[source.type[0]];
        sumpartial[0]=source.natom[0];
        nemitted=floor(source.natom[0]*(1-exp2(-tend/thalf)));//set initial numbers
        for(int i=1;i<source.NSource;i++)
        {
            thalf= isotopes.halftime[source.type[i]];
            sumpartial[i]=sumpartial[i-1]+source.natom[i];
            nemitted+=floor(source.natom[i]*(1-exp2(-tend/thalf)));
        }
        totalatom=sumpartial[source.NSource-1];
        printf("tstart is %f total possible number is %d possible emitted in the remaining time interval %d\n", tstart, totalatom, nemitted);
        ttemp=tend;
        if(nemitted>NPART/2)
        {
            ttemp=findT(0, tend, isotopes, source);
            //printf("searched time point is %f time interval is %f\n", tstart+ttemp,ttemp);
            enough=0;
        }
        printf("searched time point is %f time interval is %f\n", tstart+ttemp,ttemp);
        CUDA_CALL(cudaMemcpy(d_sumpartial,sumpartial,sizeof(unsigned int)*source.NSource,cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMemcpyToSymbol(d_time,a,sizeof(double)*NPART,0,cudaMemcpyHostToDevice));

//set position and sampling time for photons
        setPosition <<<NRAND/NTHREAD_PER_BLOCK_GPET,NTHREAD_PER_BLOCK_GPET>>> (source.NSource, totalatom, tstart, ttemp, d_natom, d_sumpartial, d_type, d_shape, d_shapecoeff, 
             d_halftime, d_decayRatio, d_coef, useprange_h);
        cudaDeviceSynchronize();

        CUDA_CALL(cudaMemcpyFromSymbol(&curemitted, d_curemitted,sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));         
        printf("currently emitted photons %d\n", (curemitted - preemitted)*2);
        CUDA_CALL(cudaMemcpy(source.natom,d_natom,sizeof(unsigned int)*source.NSource,cudaMemcpyDeviceToHost));

#if OUTPUTPSF == 2
        CUDA_CALL(cudaMemcpyFromSymbol( &(xbufferRepeat[0]), x_gBrachy,sizeof(float3)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(vxbufferRepeat[0]), vx_gBrachy, sizeof(float4)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(eventidbuffer[0]), d_eventid, sizeof(int)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(h_time[0]), d_time, sizeof(double)*NPART, 0, cudaMemcpyDeviceToHost));
        //printf("copy data finished\n");
        ofstream out1("./output/outsource.dat",ios::app|ios::binary);
        ofstream outid1("./output/idsource.dat",ios::app|ios::binary);
        ofstream outt1("./output/timesource.dat",ios::app|ios::binary);
        for(int ii=0;ii<NPART;ii++)
        {
            if(h_time[ii]>0)
            {
                out1.write((char*) &(xbufferRepeat[ii]),sizeof(float3));
                out1.write((char*) &(vxbufferRepeat[ii]),sizeof(float4));
                outid1.write((char*) &(eventidbuffer[ii]),sizeof(int));
                outt1.write((char*) &(h_time[ii]),sizeof(double));
            }
            
        }
        out1.close();
        outid1.close();
        outt1.close();
#endif

//moving time slices
        tstart+=ttemp;
        tend-=ttemp;
        
        photon<<<NRAND/512, 512>>>(NPART);
        cudaDeviceSynchronize();

#if OUTPUTPSF == 2
        CUDA_CALL(cudaMemcpyFromSymbol( &(xbufferRepeat[0]), x_gBrachy,sizeof(float3)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(vxbufferRepeat[0]), vx_gBrachy, sizeof(float4)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(eventidbuffer[0]), d_eventid, sizeof(int)*NPART, 0, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpyFromSymbol(&(h_time[0]), d_time, sizeof(double)*NPART, 0, cudaMemcpyDeviceToHost));
        //printf("copy data finished\n");
        ofstream out("./output/outphantom.dat",ios::app|ios::binary);
        ofstream outid("./output/idphantom.dat",ios::app|ios::binary);
        ofstream outt("./output/timephantom.dat",ios::app|ios::binary);
        for(int ii=0;ii<NPART;ii++)
        {
            if(h_time[ii]>0)
            {
                out.write((char*) &(xbufferRepeat[ii]),sizeof(float3));
                out.write((char*) &(vxbufferRepeat[ii]),sizeof(float4));
                outid.write((char*) &(eventidbuffer[ii]),sizeof(int));
                outt.write((char*) &(h_time[ii]),sizeof(double));
            }
            
        }
        out.close();
        outid.close();
        outt.close();
#endif       

        CUDA_CALL(cudaMemcpyToSymbol(nsstk, &zero, sizeof(int), 0, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(counts_d,temptemp,sizeof(int)*2,cudaMemcpyHostToDevice));
        
        photonde<<<1+(NPART-1)/NTHREAD_PER_BLOCK_GPET, NTHREAD_PER_BLOCK_GPET,nShared>>>(events_d,counts_d,NPART, preemitted,dens_panel, mat_panel, panelID, lengthx_panel, lengthy_panel, lengthz_panel,
                MODx_panel, MODy_panel, MODz_panel, Mspacex_panel, Mspacey_panel, Mspacez_panel,
                LSOx_panel, LSOy_panel, LSOz_panel, spacex_panel, spacey_panel, spacez_panel,
                offsetx_panel, offsety_panel, offsetz_panel, directionx_panel, directiony_panel, directionz_panel,
                UniXx_panel, UniXy_panel, UniXz_panel, UniYx_panel, UniYy_panel, UniYz_panel,
                UniZx_panel, UniZy_panel, UniZz_panel);
        cudaDeviceSynchronize();//*/   
        CUDA_CALL(cudaMemcpyFromSymbol(&nsstk_h, nsstk, sizeof(int), 0, cudaMemcpyDeviceToHost));
        printf("\nthere are %d Hits in this batch\n", nsstk_h/5);
        preemitted = curemitted;

#if OUTPUTHIT==1
        void *tempData;
        cudaMalloc( (void **) &tempData, nsstk_h*sizeof(int));
        CUDA_CALL(cudaMemcpyFromSymbol(tempData, sid, nsstk_h*sizeof(int), 0,cudaMemcpyDeviceToDevice));
        outputData(tempData, nsstk_h*sizeof(int), "./output/HitsID.dat", "ab");
        cudaFree(tempData);

        void *tempData2;
        CUDA_CALL(cudaMalloc( (void **) &tempData2, sizeof(float)*nsstk_h));
        CUDA_CALL(cudaMemcpyFromSymbol(tempData2, sf, sizeof(float)*nsstk_h, 0,cudaMemcpyDeviceToDevice));
        outputData(tempData2,sizeof(float)*(nsstk_h), "./output/Hits.dat", "ab");
        cudaFree(tempData2);
#endif
        int counts=0;
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after adder is "<<counts<<endl;
        outevents(&counts,events_d,"./output/adder.dat");

//insert proper digitizer module in the following part
//the module that can change the number of events should be followed by a sort
        //energy blur and spatial blur
        blur<<<NRAND/512,512>>>(counts,events_d,blurpolicy, Eref, Rref, Eslope, Sblur);
        cudaDeviceSynchronize();
        //outevents(&counts,events_d,"./output/blur.dat");

        //energy window
        energywindow<<<NRAND/512,512>>>(counts_d,events_d, counts, Eth,2000000);
        cudaDeviceSynchronize();
        //quicksort_d(events_d,0,counts,3);
        quicksort_h(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after thresholder is "<<counts<<endl;
        //outevents(&counts,events_d,"./output/thresholder1.dat");

        //deadtime part
        if(dlevel!=3)
        {
            setSitenum<<<NRAND/512,512>>>(counts,events_d,dlevel);
            cudaDeviceSynchronize();
            printf("set site number ok\n");
        } 
        orderevents(&counts,events_d);//make events globally ordered by site number, and then ordered by flight time in each volume
        deadtime<<<NRAND/512,512>>>(counts_d,events_d, counts, dtime, dtype);
        cudaDeviceSynchronize();
        cout<<"deadtime is ok\n";
        //quicksort_d(events_d,0,counts,3);
        quicksort_h(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after deadtime is "<<counts<<endl;
        //outevents(&counts,events_d,"./output/deadtime.dat");//*/

        energywindow<<<NRAND/512,512>>>(counts_d,events_d, counts, Ewinmin,Ewinmax);
        cudaDeviceSynchronize();
        //quicksort_d(events_d,0,counts,3);
        quicksort_h(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of singles is "<<counts<<endl;
        outevents(counts_d,events_d,"./output/singles.dat");

        printf("finish %d run\n", epoch);
        if(enough ) break;
    }
    cudaFree(events_d);
    cudaFree(counts_d);

    end_time = clock();
    time_diff = ((float)end_time - (float)start_time)/CLOCKS_PER_SEC;
    printf("\n\n****************************************\n");
    printf("Simulation time: %f s.\n\n",time_diff);
    printf("****************************************\n\n\n");
}

float findT(float tstart, float tend, Isotopes isotopes, Source source)
{
    float tmid= (tstart+tend)*0.5;
    float thalf;
    int nemitted=0;
    for(int i=0;i<source.NSource;i++)
    {
        thalf= isotopes.halftime[source.type[i]];
        nemitted += floor(source.natom[i]*(1-exp2(-tmid/thalf)));
    }
    if(nemitted>0.98*NPART*0.5) tmid= findT(tstart,  tmid, isotopes, source);
    else if(nemitted<0.95*NPART*0.5) tmid= findT(tmid, tend, isotopes, source);
    else return tmid;
}

#endif
