#ifndef __GBRACHYINTERNAL_H__
#define __GBRACHYINTERNAL_H__
#include "constants.h"


float Eref, Rref, Eslope, Sblur, Eth, Ewinmin, Ewinmax, dtime;
int blurpolicy, dlevel, dtype, useprange_h;
float emax, eminph;


// common variable group for panel geometry
float *dens_panel;
int *mat_panel;
int *panelID;
float *lengthx_panel, *lengthy_panel, *lengthz_panel;
float *MODx_panel, *MODy_panel, *MODz_panel;
float *Mspacex_panel, *Mspacey_panel, *Mspacez_panel;
float *LSOx_panel, *LSOy_panel, *LSOz_panel;
float *spacex_panel, *spacey_panel, *spacez_panel;
float *offsetx_panel, *offsety_panel, *offsetz_panel;
float *directionx_panel, *directiony_panel, *directionz_panel;
float *UniXx_panel, *UniXy_panel, *UniXz_panel;
float *UniYx_panel, *UniYy_panel, *UniYz_panel;
float *UniZx_panel, *UniZy_panel, *UniZz_panel;

	


//fixed parameters

float eabsph_h;
float elaph_h[NLAPH],lamph_h[NLAPH*MAXMAT],lampha_h[NLAPH*MAXMAT],lamphb_h[NLAPH*MAXMAT],
	lamphc_h[NLAPH*MAXMAT],lamphd_h[NLAPH*MAXMAT];

float idleph_h;
cudaArray *lamph;

float idlecp_h;
float ecmpt_h[NCMPT],compt_h[NCMPT*MAXMAT];
cudaArray *compt;

float idcpcm_h, idecm_h;

float mucmpt_h[NCPCM*NECM*MAXMAT];
cudaArray* sArray;

float idlepe_h;
float ephte_h[NPHTE],phote_h[NPHTE*MAXMAT];
cudaArray *phote;

float idlerl_h;
float erayl_h[NRAYL],rayle_h[NRAYL*MAXMAT];
cudaArray *rayle;

float idcprl_h, iderl_h;
float murayl_h[NCPRL*NERL*MAXMAT];
cudaArray* fArray;

//woodcock
float idlewk_h, wcke0_h;
float woock_h[NWCK];
cudaArray *woock;
cudaArray *woockde;

#endif
