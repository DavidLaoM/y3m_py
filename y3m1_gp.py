# -*- coding = utf-8 -*-
"""
Simulation of a 110 mM glucose perturbation at a dilution rate of 0.1 h-1.
"""

import tellurium as te
import roadrunner
import antimony
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile

r = te.loada ('''
model feedback()
  // Reactions:
  vGLT: -> GLCi;                                    p_GLT_VmGLT * (f_GLCo-GLCi / p_GLT_KeqGLT) / (p_GLT_KmGLTGLCo * (1+f_GLCo/p_GLT_KmGLTGLCo+GLCi / p_GLT_KmGLTGLCi + 0.91 * f_GLCo * GLCi / (p_GLT_KmGLTGLCi * p_GLT_KmGLTGLCo))); 
  vGLK: GLCi + ATP -> G6P + ADP;                    p_HXK_ExprsCor * (((p_HXK1_kcat * (f_HXK1+f_HXK2)) / (p_HXK1_Katp * p_HXK1_Kglc) * (ATP * GLCi-((ADP * G6P) / p_HXK1_Keq))) / ((1+ATP / p_HXK1_Katp+ADP / p_HXK1_Kadp) * (1 + GLCi / p_HXK1_Kglc+G6P / p_HXK1_Kg6p+T6P / p_HXK1_Kt6p))); 
  vPGI: G6P -> F6P;                                 p_PGI_ExprsCor * ((((p_PGI1_kcat*f_PGI1) / p_PGI1_Kg6p) * (G6P-(F6P / p_PGI1_Keq))) / (1+G6P / p_PGI1_Kg6p+1+F6P / p_PGI1_Kf6p-1)); 
  vPFK: F6P + ATP -> F16BP + ADP;                   p_PFK_ExprsCor * ((p_PFK_kcat * f_PFK * p_PFK_gR * (F6P / p_PFK_Kf6p) * (ATP / p_PFK_Katp) * (1+(F6P / p_PFK_Kf6p)+(ATP / p_PFK_Katp)+p_PFK_gR * ((F6P / p_PFK_Kf6p) * (ATP / p_PFK_Katp)))) / ((1+F6P / p_PFK_Kf6p+ATP / p_PFK_Katp+(p_PFK_gR * (F6P / p_PFK_Kf6p) * (ATP / p_PFK_Katp))) ^ 2 + p_PFK_L * ((1+p_PFK_Ciatp * (ATP / p_PFK_Kiatp)) / (1+ATP / p_PFK_Kiatp)) ^ 2 * ((1+p_PFK_Camp * (AMP / p_PFK_Kamp)) / (1+AMP / p_PFK_Kamp)) ^ 2 * ((1+((p_PFK_Cf26bp*p_PFK_F26BP) / (p_PFK_Kf26bp))+((p_PFK_Cf16bp * F16BP) / (p_PFK_Kf16bp))) / (1+(p_PFK_F26BP / p_PFK_Kf26bp)+(F16BP / p_PFK_Kf16bp))) ^ 2 * (1+p_PFK_Catp * (ATP / p_PFK_Katp)) ^ 2)); 
  vALD: F16BP -> GLYCERAL3P + DHAP;                 p_FBA_ExprsCor * (((p_FBA1_kcat * f_FBA1) / p_FBA1_Kf16bp * (F16BP-(GLYCERAL3P * DHAP) / p_FBA1_Keq)) / (1+F16BP / p_FBA1_Kf16bp+(1+GLYCERAL3P / p_FBA1_Kglyceral3p) * (1+DHAP / p_FBA1_Kdhap)-1));
  vTPI: DHAP -> GLYCERAL3P;                         (((p_TPI1_kcat*f_TPI1) / p_TPI1_Kdhap * (DHAP-GLYCERAL3P / p_TPI1_Keq)) / (1+DHAP / p_TPI1_Kdhap+1+GLYCERAL3P / p_TPI1_Kglyceral3p-1));
  vGAPDH: GLYCERAL3P + NAD + PHOS -> BPG + NADH;    p_GAPDH_ExprsCor * ((((p_TDH1_kcat * (f_TDH1+f_TDH2+f_TDH3)) / (p_TDH1_Kglyceral3p * p_TDH1_Knad * p_TDH1_Kpi)) * (GLYCERAL3P * NAD * PHOS-(BPG * NADH) / p_TDH1_Keq)) / ((1+GLYCERAL3P / p_TDH1_Kglyceral3p) * (1+NAD / p_TDH1_Knad) * (1+ PHOS / p_TDH1_Kpi)+(1+BPG / p_TDH1_Kglycerate13bp) * (1+NADH / p_TDH1_Knadh)-1));
  vPGK: BPG + ADP -> P3G + ATP;                     p_PGK_ExprsCor * p_PGK_VmPGK * ((p_PGK_KeqPGK * BPG * ADP)-ATP * P3G) / (p_PGK_KmPGKATP*p_PGK_KmPGKP3G * (1+ADP / p_PGK_KmPGKADP + ATP / p_PGK_KmPGKATP) * (1+BPG/p_PGK_KmPGKBPG+P3G/p_PGK_KmPGKP3G));
  vPGM: P3G -> P2G;                                 p_PGM_ExprsCor * ((((p_GPM1_kcat * (f_GPM1+f_GPM2+f_GPM3)) / p_GPM1_K3pg) * (P3G-P2G / p_GPM1_Keq)) / (1+P3G / p_GPM1_K3pg+1+P2G / p_GPM1_K2pg-1));
  vENO: P2G -> PEP;                                 p_ENO_ExprsCor * ((((p_ENO1_kcat * (f_ENO1+f_ENO2)) / p_ENO1_K2pg) * (P2G-PEP / p_ENO1_Keq)) / (1+P2G / p_ENO1_K2pg+1+PEP / p_ENO1_Kpep-1));
  vPYK: PEP + ADP -> PYR + ATP;                     p_PYK_ExprsCor * ((((p_PYK1_kcat * (f_PYK1+f_PYK2)) / (p_PYK1_Kadp * p_PYK1_Kpep) * ADP * PEP) / ((1+ADP / p_PYK1_Kadp) * (1+PEP / p_PYK1_Kpep))) * ((PEP / p_PYK1_Kpep+1) ^ p_PYK1_hill / (p_PYK1_L * ((ATP / p_PYK1_Katp+1) / (F16BP / p_PYK1_Kf16bp+1)) ^ p_PYK1_hill+(PEP / p_PYK1_Kpep+1) ^ p_PYK1_hill)));
  vPDC: PYR -> ACE;                                 p_PDC_ExprsCor * ((p_PDC1_kcat * (f_PDC1) * (PYR / p_PDC1_Kpyr) ^ p_PDC1_hill) / (1+(PYR / p_PDC1_Kpyr) ^ p_PDC1_hill+ PHOS / p_PDC1_Kpi));
  vADH: ACE + NADH -> ETOH + NAD;                   -p_ADH_ExprsCor * (p_ADH_VmADH / (p_ADH_KiADHNAD * p_ADH_KmADHETOH) * (NAD * ETOH-NADH * ACE / p_ADH_KeqADH) / (1+NAD / p_ADH_KiADHNAD+p_ADH_KmADHNAD * ETOH / (p_ADH_KiADHNAD*p_ADH_KmADHETOH)+p_ADH_KmADHNADH * ACE / (p_ADH_KiADHNADH * p_ADH_KmADHACE) + NADH / p_ADH_KiADHNADH+NAD * ETOH / (p_ADH_KiADHNAD * p_ADH_KmADHETOH)+p_ADH_KmADHNADH * NAD * ACE / (p_ADH_KiADHNAD * p_ADH_KiADHNADH * p_ADH_KmADHACE) + p_ADH_KmADHNAD * ETOH * NADH / (p_ADH_KiADHNAD * p_ADH_KmADHETOH * p_ADH_KiADHNADH)+NADH * ACE / (p_ADH_KiADHNADH * p_ADH_KmADHACE) + NAD * ETOH * ACE / (p_ADH_KiADHNAD*p_ADH_KmADHETOH * p_ADH_KiADHACE)+ETOH * NADH * ACE / (p_ADH_KiADHETOH*p_ADH_KiADHNADH*p_ADH_KmADHACE)));
  vG3PDH: NADH + DHAP -> NAD + GLYC3P;              ((((p_GPD1_kcat * f_GPD1) / (p_GPD1_Kdhap * p_GPD1_Knadh)) * (DHAP * NADH-(GLYC3P * NAD) / p_GPD1_Keq)) / ((1+F16BP / p_GPD1_Kf16bp+ATP / p_GPD1_Katp+ADP / p_GPD1_Kadp) * (1+DHAP / p_GPD1_Kdhap+GLYC3P / p_GPD1_Kglyc3p) * (1+NADH / p_GPD1_Knadh+NAD / p_GPD1_Knad))); 
  vHOR2: GLYC3P -> GLYCEROL + PHOS;                 (((p_HOR2_kcat * f_HOR2) / p_HOR2_Kglyc3p * GLYC3P) / ((1+ PHOS / p_HOR2_Kpi) * (1+GLYC3P / p_HOR2_Kglyc3p)));
  vPGM1: G1P -> G6P;                                (((p_PGM1_kcat * (f_PGM1 + f_PGM2 + f_PGM3)) / p_PGM1_Kg1p * (G1P-G6P / p_PGM1_Keq)) / (1 + G1P / p_PGM1_Kg1p + G6P / p_PGM1_Kg6p)); # 0;
  vTPS1: G6P + G1P + ATP -> T6P + ADP + 2 PHOS;     (F6P / (F6P + p_TPS1_KmF6P)) * (((p_TPS1_kcat * f_TPS1) / (p_TPS1_Kg6p * p_TPS1_Kudp_glc) * G6P * UDP_GLC / ((1 + G6P / p_TPS1_Kg6p) * (1 + UDP_GLC / p_TPS1_Kudp_glc) * (1 + PHOS / p_TPS1_Kpi) ))); # 0;
  vTPS2: T6P -> TRE + PHOS;                         (((p_TPS2_kcat * f_TPS2) * T6P * PHOS) / ((p_TPS2_Kt6p * p_TPS2_Kpi) + (p_TPS2_Kt6p + T6P) * PHOS)); # 0;
  vNTH1: TRE -> 2 GLCi;                             (((p_NTH1_kcat * f_NTH1) / p_NTH1_Ktre * TRE) / (1 + TRE / p_NTH1_Ktre)); # 0;
  vVacPi: -> PHOS;                                  p_vacuolePi_k * (p_vacuolePi_steadyStatePi - PHOS);
  vADK1: 2 ADP -> ATP + AMP;                        p_ADK1_k * ((ADP * ADP)-(AMP * ATP) / p_ADK1_Keq);
  vmitoNADH: NADH -> NAD;                           p_mitoNADHVmax * (NADH / (NADH + p_mitoNADHKm));
  vAmd1: AMP -> IMP;                                (p_Amd1_Vmax * AMP) / (p_Amd1_K50 * (1+ PHOS / p_Amd1_Kpi) / (ATP / p_Amd1_Katp + 1) + AMP);
  vAde1312: IMP -> AMP;                             IMP * p_Ade13_Ade12_k;
  vIsn1: IMP -> INO + PHOS;                         IMP * p_Isn1_k; 
  vPnp1: INO -> HYP;                                INO * p_Pnp1_k;
  vHpt1: HYP + PHOS -> IMP;                         HYP * p_Hpt1_k;
  vETOHt: ETOH -> ;                                 p_kETOHtransport*(ETOH-f_ETOH_e);
  vGLYCt: GLYCEROL -> ;                             p_GlycerolTransport * (GLYCEROL-f_GLYCEROL_e); 
  vATPase: ATP -> ADP + PHOS;                       p_ATPase_ratio * ATP  /  ADP;
  vmito: ADP + PHOS -> ATP;                         p_mitoVmax * ADP / (ADP + p_mitoADPKm) * ( PHOS / ( PHOS + p_mitoPiKm));
  vsinkG6P: PHOS -> G6P;                            poly_sinkG6P * (G6P / (G6P + km_sinkG6P));
  vsinkF6P: PHOS -> F6P;                            poly_sinkF6P * (F6P / (F6P + km_sinkF6P));
  vsinkGAP: PHOS -> GLYCERAL3P;                     poly_sinkGAP * (GLYCERAL3P / (GLYCERAL3P + km_sinkGAP)); 
  vsinkP3G: PHOS -> P3G;                            poly_sinkP3G * (P3G / (P3G + km_sinkP3G));
  vsinkPEP: PHOS -> PEP;                            poly_sinkPEP * (PEP / (PEP + km_sinkPEP));
  vsinkPYR: -> PYR;                                 poly_sinkPYR * (PYR / (PYR + km_sinkPYR));
  vsinkACE: -> ACE;                                 poly_sinkACE * (ACE / (ACE + km_sinkACE));
  
  // Species initializations:
  ACE = 0.0400;
  BPG = 0;
  F16BP = 0.2050;
  F6P = 0.6600;
  G6P = 2.4950;
  GLCi = 0.2000;
  NAD = 1.5794;
  NADH = 0.0106;
  ATP = 2.2150;
  P2G = 0.3410;
  P3G = 3.0150;
  PEP = 1.3400; 
  PYR = 0.3250; 
  GLYCERAL3P = 0.0100; 
  ADP = 0.5300; 
  AMP = 0.0347; 
  DHAP = 0.1815; 
  GLYC3P = 0.0300; 
  GLYCEROL = 0.1000; 
  ETOH = 10.0000; 
  G1P = 0.1300;
  TRE = 56.1303;
  T6P = 0.1000; 
  PHOS = 10;
  IMP = 0.1000; 
  INO = 0.1000;
  HYP = 1.5000; 
  ETOHec = 0; 
  GLYCec = 0.0900; 

  // Variable initialization:
  p_HXK_ExprsCor = 1;
  p_PGI_ExprsCor = 1;
  p_PFK_ExprsCor = 1;
  p_FBA_ExprsCor = 1;
  p_TPI_ExprsCor = 1;
  p_GAPDH_ExprsCor = 1;
  p_PGK_ExprsCor = 1;
  p_PGM_ExprsCor = 1;
  p_ENO_ExprsCor = 1;
  p_PYK_ExprsCor = 1;
  p_PDC_ExprsCor = 1;
  p_ADH_ExprsCor = 1;
  p_GLT_KeqGLT = 1; 
  p_GLT_KmGLTGLCi = 1.0078; 
  p_GLT_KmGLTGLCo = 1.0078; 
  p_GLT_VmGLT = 8.1327;
  p_HXK1_Kadp = 0.3492;
  p_HXK1_Katp = 0.0931;
  p_HXK1_Keq = 3.7213e+03;
  p_HXK1_Kg6p = 34.7029;
  p_HXK1_Kglc = 0.3483;
  p_HXK1_Kt6p = 0.0073;
  p_HXK1_kcat = 6.2548;
  p_PFK_Camp = 0.0287;
  p_PFK_Catp = 1.2822;
  p_PFK_Cf16bp = 2.3638;
  p_PFK_Cf26bp = 0.0283;
  p_PFK_Ciatp = 40.3824;
  p_PFK_Kamp = 0.0100;
  p_PFK_Katp = 1.9971;
  p_PFK_Kf16bp = 0.0437;
  p_PFK_Kf26bp = 0.0012;
  p_PFK_Kf6p = 0.9166;
  p_PFK_Kiatp = 4.9332;
  p_PFK_L = 1.3886;
  p_PFK_gR = 1.8127;
  p_PFK_kcat = 8.7826;
  p_PFK_F26BP = 1e-3;  
  p_PGI1_Keq = 0.9564; #0.956375672911768; #0.9564; # number of decimals tis important
  p_PGI1_Kf6p = 7.2433; #7.243331730145231; #7.2433; # number of decimals tis important
  p_PGI1_Kg6p = 33.0689; #33.068946195264843; #33.0689; # number of decimals tis important
  p_PGI1_kcat = 2.3215; #2.321459895423278; #2.3215; # number of decimals tis important
  p_FBA1_Kdhap = 0.0300;
  p_FBA1_Keq = 0.1223;
  p_FBA1_Kf16bp = 0.6872;
  p_FBA1_Kglyceral3p = 3.5582;
  p_FBA1_kcat = 4.4067;
  p_TPI1_Kdhap = 205.9964;
  p_TPI1_Keq = 0.0515;
  p_TPI1_Kglyceral3p = 8.8483;
  p_TPI1_kcat = 16.1694;
  p_GPM1_K2pg = 0.0750;
  p_GPM1_K3pg = 1.4151;
  p_GPM1_Keq = 0.1193;
  p_GPM1_kcat = 11.3652;
  p_ENO1_K2pg = 0.0567;
  p_ENO1_Keq = 4.3589;
  p_ENO1_Kpep = 0.4831;
  p_ENO1_kcat = 3.3018;
  p_PYK1_Kadp = 0.2430;
  p_PYK1_Katp = 9.3000;
  p_PYK1_Kf16bp = 0.1732;
  p_PYK1_Kpep = 0.2810;
  p_PYK1_L = 60000;
  p_PYK1_hill = 4;
  p_PYK1_kcat = 9.3167;
  p_TDH1_Keq = 0.0054;
  p_TDH1_Kglyceral3p = 4.5953;
  p_TDH1_Kglycerate13bp = 0.9076;
  p_TDH1_Knad = 1.1775;
  p_TDH1_Knadh = 0.0419;
  p_TDH1_Kpi = 0.7731;
  p_TDH1_kcat = 78.6422;
  p_PGK_KeqPGK = 3.2348e+03;
  p_PGK_KmPGKADP = 0.2064;
  p_PGK_KmPGKATP = 0.2859;
  p_PGK_KmPGKBPG = 0.0031;
  p_PGK_KmPGKP3G = 0.4759;
  p_PGK_VmPGK = 55.1626;
  p_GPD1_Kadp = 1.1069;
  p_GPD1_Katp = 0.5573;
  p_GPD1_Kdhap = 2.7041;
  p_GPD1_Keq = 1.0266e+04;
  p_GPD1_Kf16bp = 12.7519;
  p_GPD1_Kglyc3p = 3.2278;
  p_GPD1_Knad = 0.6902;
  p_GPD1_Knadh = 0.0322;
  p_GPD1_kcat = 1.7064;
  p_PDC1_Kpi = 9.4294;
  p_PDC1_Kpyr = 12.9680;
  p_PDC1_hill = 0.7242;
  p_PDC1_kcat = 8.3613;
  p_ADH_KeqADH = 6.8487e-05; 
  p_ADH_KiADHACE = 0.6431; 
  p_ADH_KiADHETOH = 59.6935; 
  p_ADH_KiADHNAD = 0.9677; 
  p_ADH_KiADHNADH = 0.0316; 
  p_ADH_KmADHACE = 1.1322; 
  p_ADH_KmADHETOH = 4.8970; 
  p_ADH_KmADHNAD = 0.1534; 
  p_ADH_KmADHNADH = 0.1208; 
  p_ADH_VmADH = 13.2581; 
  p_HOR2_Kglyc3p = 2.5844;
  p_HOR2_Kpi = 2.5491;
  p_HOR2_kcat = 1.2748;
  p_PGM1_Keq = 21.3955;
  p_PGM1_Kg1p = 0.0653;
  p_PGM1_Kg6p = 0.0324;
  p_PGM1_kcat = 8.4574;
  p_TPS1_Kg6p = 4.5359; # 4.535897437530080; # 4.5359; 
  p_TPS1_Kudp_glc = 0.1268; # 0.126815908482557; # 0.1268; 
  p_TPS1_kcat = 9.6164e+03; # 9.616420586003847e+03; # 9.6164e+03; 
  p_TPS1_Kpi = 0.7890; # 0.789017889574395; # 0.7890; 
  p_TPS1_KmF6P = 1.5631; # 1.563120474207156; # 1.5631; 
  p_TPS2_Kt6p = 0.3686;
  p_TPS2_kcat = 28.4097;
  p_TPS2_Kpi = 0.7023;
  p_NTH1_Ktre = 2.1087;
  p_NTH1_kcat = 4.5132;
  p_GlycerolTransport = 0.1001;
  p_kETOHtransport = 0.0328;
  p_VmaxACE = 0.1456;
  p_KmACE = 0.3315;
  p_PDH_Vmax = 0.5284;
  p_PDH_n = 6.3869;
  p_PDH_K50 = 0.3923;
  p_vacuolePi_k = 0.1699;
  p_vacuolePi_steadyStatePi = 10;
  p_mitoNADHVmax = 0.2401;
  p_mitoNADHKm = 1.0000e-03;
  p_mitoVmax = 1.6034;
  p_mitoADPKm = 0.3394;
  p_mitoPiKm = 0.4568;
  p_ATPaseK = 0.0346;
  p_Amd1_K50 = 10.9184;
  p_Amd1_Kpi = 1.6184e+03;
  p_Amd1_Katp = 5000;
  p_ATPase_Katp = 0;
  p_ATPase_ratio2 = 1.8211;
  p_ADK1_k = 77.3163;
  p_ADK1_Keq = 0.2676;
  poly_sinkACE = -0.034836166800000; 
  poly_sinkF6P = 0.024574614000000; 
  poly_sinkG6P = -0.077853600000000; 
  poly_sinkGAP = 0.012626909700000; 
  poly_sinkP3G = -0.007881000000000; 
  poly_sinkPEP = -0.007607000000000; 
  poly_sinkPYR = -0.161328300000000; 
  km_sinkACE = 1e-04;
  km_sinkF6P = 1e-04; 
  km_sinkG6P = 1e-02;
  km_sinkGAP = 5e-04;
  km_sinkP3G = 1e-03;
  km_sinkPEP = 1e-03;
  km_sinkPYR = 1e-03;
  f_GLK1 = 0; 
  f_HXK1 = 1; 
  f_HXK2 = 0; 
  f_PGI1 = 1; 
  f_PFK = 1; 
  f_FBA1 = 1; 
  f_GPD1 = 1; 
  f_GPD2 = 0; 
  f_TDH1 = 1; 
  f_TDH2 = 0; 
  f_TDH3 = 0; 
  f_PGK1 = 0.1320; 
  f_GPM1 = 1; 
  f_GPM2 = 0; 
  f_GPM3 = 0; 
  f_ENO1 = 1; 
  f_ENO2 = 0; 
  f_PYK1 = 1; 
  f_PYK2 = 0; 
  f_PDC1 = 0.5290; 
  f_PDC5 = 0.0059; 
  f_PDC6 = 0.0034; 
  f_ADH1 = 0.0933; 
  f_ADH2 = 0; 
  f_ADH3 = 0.0019; 
  f_ADH4 = 0.0359; 
  f_ADH5 = 0.0023; 
  f_ADH6 = 0.0171; 
  f_ADH7 = 0; 
  f_TPI1 = 1; 
  f_HOR2 = 1; 
  f_RHR2 = 0; 
  f_PGM1 = 1; 
  f_PGM2 = 0; 
  f_PGM3 = 0; 
  f_UGP1 = 3.1000e-04; 
  f_TPS2 = 0.0013; 
  f_NTH1 = 0.0020; 
  f_TPS1 = 0.0014; # 0.00145;  
  f_CO2 = 1; 
  f_GLY = 0.1500; 
  f_Glyc = 0; 
  f_SUCC = 0; 
  f_Trh = 0; 
  f_X = 0; 
  f_GLYCEROL_e = 0; 
  f_ETOH_e = 0; 
  f_TRE_e = 0; 
  UDP_GLC = 0.07;
  # adjustments upon glucose perturbation
  # f_GLCo = 0.1800; # increase to 110 mM GLCout upon perturbation
  f_GLCo := piecewise(110, time > 3000, 0.1800)  
  # p_Amd1_Vmax = 0; # 9.8464; # inosine salvage pathway active upon perturbation
  # p_Ade13_Ade12_k = 0; # 0.6298;
  # p_Isn1_k = 0; # 0.3654;
  # p_Pnp1_k = 0; # 0.0149;
  # p_Hpt1_k = 0; # 0.0112;
  p_Amd1_Vmax := piecewise(9.8464, time > 3000, 0)
  p_Ade13_Ade12_k := piecewise(0.6298, time > 3000, 0)
  p_Isn1_k := piecewise(0.3654, time > 3000, 0)
  p_Pnp1_k := piecewise(0.0149, time > 3000, 0)
  p_Hpt1_k := piecewise(0.0112, time > 3000, 0) 
  # p_ATPase_ratio = 0.23265; # adjusted ATPase activity upon perturbation
  p_ATPase_ratio := piecewise(0.205, time > 3000, 0.23265) 
  E1: at ((time > 3000) && (time < 3001)): PHOS=25; # increase PHOS realease this time by changing parameters
  
end''')
# f_sbml = tempfile.NamedTemporaryFile(suffix=".xml")
# r.exportToSBML(f_sbml.name)
# r.exportToSBML(f_sbml.name, current=False)


# simulation
result = r.simulate(0, 3340, 3341)
colnames = [ "time","[GLCi]","[ATP]","[G6P]","[ADP]","[T6P]","[F6P]","[F16BP]","[AMP]","[GLYCERAL3P]","[DHAP]","[NAD]","[PHOS]","[BPG]","[NADH]","[P3G]","[P2G]","[PEP]","[PYR]","[ACE]",  "[ETOH]",  "[GLYC3P]", "[GLYCEROL]", "[G1P]",   "[TRE]",     "[IMP]",    "[INO]",   "[HYP]"]
# plot
r.plot(result)
# # 
# temp_id = list(range(1,28,1))
# print(result[-1,temp_id])

# # export to SBML
# f_sbml = tempfile.NamedTemporaryFile(suffix=".xml")
# r.exportToSBML(f_sbml.name)

# str_sbml = r.getSBML()
# print(str_sbml)
