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
from scipy.interpolate import CubicSpline

r = te.loada ('''
model feedback()
  // Reactions:
  vGLT: 0.007278 GLCec -> GLCi;                     p_GLT_VmGLT * ((GLCec-Csmin)-GLCi / p_GLT_KeqGLT) / (p_GLT_KmGLTGLCo * (1+(GLCec-Csmin)/p_GLT_KmGLTGLCo+GLCi / p_GLT_KmGLTGLCi + 0.91 * (GLCec-Csmin) * GLCi / (p_GLT_KmGLTGLCi * p_GLT_KmGLTGLCo))); 
  vGLK: GLCi + ATP -> G6P + ADP;                    p_HXK_ExprsCor * (((p_HXK1_kcat * (f_HXK1+f_HXK2)) / (p_HXK1_Katp * p_HXK1_Kglc) * (ATP * GLCi-((ADP * G6P) / p_HXK1_Keq))) / ((1+ATP / p_HXK1_Katp+ADP / p_HXK1_Kadp) * (1 + GLCi / p_HXK1_Kglc+G6P / p_HXK1_Kg6p + T6P / p_HXK1_Kt6p))); 
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
  vUGP: G1P -> UDP_GLC;                             (p_UGP_kcat * f_UGP) * ( ( UTP / p_UGP_Kutp ) * ( G1P / p_UGP_Kg1p ) )  / ( p_UGP_Ki_utp / p_UGP_Kutp + UTP / p_UGP_Kutp + G1P / p_UGP_Kg1p + ( UTP / p_UGP_Kutp ) * ( G1P / p_UGP_Kg1p ) + ( p_UGP_Ki_utp / p_UGP_Kutp ) * ( UDP_GLC / p_UGP_Ki_udp_glc ) + ( G1P / p_UGP_Kg1p ) * ( UDP_GLC / p_UGP_Ki_udp_glc ) );
  vTPS1: G6P + UDP_GLC + ATP -> T6P + ADP + 2 PHOS; (F6P / (F6P + p_TPS1_KmF6P)) * (((p_TPS1_kcat * f_TPS1) / (p_TPS1_Kg6p * p_TPS1_Kudp_glc) * G6P * UDP_GLC / ((1 + G6P / p_TPS1_Kg6p) * (1 + UDP_GLC / p_TPS1_Kudp_glc) * (1 + PHOS / p_TPS1_Kpi) ))); # 0;
  vTPS2: T6P -> TRE + PHOS;                         (((p_TPS2_kcat * f_TPS2) * T6P * PHOS) / ((p_TPS2_Kt6p * p_TPS2_Kpi) + (p_TPS2_Kt6p + T6P) * PHOS)); # 0;
  vNTH1: TRE -> 2 GLCi;                             (((p_NTH1_kcat * f_NTH1) / p_NTH1_Ktre * TRE) / (1 + TRE / p_NTH1_Ktre)); # 0;

  vVacPi: -> PHOS;                                  p_vacuolePi_k * (p_vacuolePi_steadyStatePi- PHOS);
  vADK1: 2 ADP -> ATP + AMP;                        p_ADK1_k * ((ADP * ADP)-(AMP * ATP) / p_ADK1_Keq);
  vmitoNADH: NADH -> NAD;                           p_mitoNADHVmax * (NADH / (NADH + p_mitoNADHKm));
  vAmd1: AMP -> IMP;                                0;
  vAde1312: IMP -> AMP;                             0;
  vIsn1: IMP -> INO + PHOS;                         0; 
  vPnp1: INO -> HYP;                                0;
  vHpt1: HYP + PHOS -> IMP;                         0;
  vETOHt: ETOH -> 0.007278 ETOHec;                  p_kETOHtransport * (ETOH - ETOHec);
  vGLYCt: GLYCEROL -> 0.007278 GLYCEROLec;          p_GlycerolTransport * (GLYCEROL - GLYCEROLec); 
  vATPase: ATP -> ADP + PHOS;                       p_ATPaseK * ATP  /  ADP;
  vmito: ADP + PHOS -> ATP;                         p_mitoVmax * ADP / (ADP + p_mitoADPKm) * ( PHOS / ( PHOS + p_mitoPiKm));
  vsinkG6P: -> G6P;                                 poly_sinkG6P * (G6P / (G6P + km_sinkG6P)); # ! phos pool in matlab
  vsinkF6P: -> F6P;                                 poly_sinkF6P * (F6P / (F6P + km_sinkF6P)); # ! phos pool in matlab
  vsinkGAP: -> GLYCERAL3P;                          poly_sinkGAP * (GLYCERAL3P / (GLYCERAL3P + km_sinkGAP));  # ! phos pool in matlab
  vsinkP3G: -> P3G;                                 poly_sinkP3G * (P3G / (P3G + km_sinkP3G)); # ! phos pool in matlab
  vsinkPEP: -> PEP;                                 poly_sinkPEP * (PEP / (PEP + km_sinkPEP)); # ! phos pool in matlab 
  vsinkPYR: -> PYR;                                 poly_sinkPYR * (PYR / (PYR + km_sinkPYR));
  vsinkACE: -> ACE;                                 poly_sinkACE * (ACE / (ACE + km_sinkACE));

  vATH1ec: 0.007278 TREec -> 0.014556 GLCec;        (p_ATH1_kcat_ec * f_ATH1ec) * ( (TREec / p_ATH1_Ktre_ec) / ( 1+(TREec / p_ATH1_Ktre_ec) + (T6P / p_ATH1_Kt6p_ec) ) );
  vATH1vac: TREvac -> 2 GLCi;                       (p_ATH1_kcat * f_ATH1vac) * ( (TREvac / p_ATH1_Ktre) / ( 1+(TREvac / p_ATH1_Ktre) + (T6P / p_ATH1_Kt6p) ) );
  vAGT1: TRE -> 0.007278 TREec;                     (p_AGT1_kcat * f_AGT1) * (1 / p_AGT1_Ktre) * ( TRE - TREec / p_AGT1_Keq ) / ( 1+TRE / p_AGT1_Ktre + TREec / p_AGT1_Ktre_ec + (UDP_GLC / p_AGT1_Ki) ) ;
  vvacuoleT: TRE -> TREvac;                         p_vacuoleT_vmax * (1 / p_vacuoleT_Ktre) * ( TRE - TREvac / p_vacuoleT_Keq) / ( 1 + TRE / p_vacuoleT_Ktre + TREvac / p_vacuoleT_Ktre);
  
  vglycSynth_FF: UDP_GLC -> Glycogen_cyt;           temp_v_glycSynth * UDP_GLC / (UDP_GLC + 0.0001);
  vglycDeg_FF: Glycogen_cyt + 0.5 PHOS -> GLCi;     temp_v_glycDeg * Glycogen_cyt / (Glycogen_cyt + 0.0001);
  vglycSynth_preFF: UDP_GLC -> Glycogen_cyt;        glycSynth_K * UDP_GLC;
  vglycDeg_preFF: Glycogen_cyt + 0.5 PHOS -> GLCi;  glycDeg_K * Glycogen_cyt;
  
  Volinc: -> Vbroth;                                Fin; 
  Voldec: Vbroth -> ;                               Fout; 
  FinGlucose: -> GLCec;                             Fin * GLCin / Vbroth; 
  FoutEthanol: ETOHec -> ;                          Fout * ETOHec / Vbroth; 
  FoutGlycerol: GLYCEROLec -> ;                     Fout * GLYCEROLec / Vbroth; 
  FoutGlucose: GLCec -> ;                           Fout * GLCec / Vbroth; 
  FoutTrehalose: TREec -> ;                         Fout * TREec / Vbroth
  
  
  // Species initializations:
  ACE = 0.0200;
  BPG = 0;
  F16BP = 0.1150;
  F6P = 0.7500;
  G6P = 2.7100; 
  GLCi = 2.7850; 
  NAD =  0.9150; 
  NADH = 0.2350; 
  ATP = 4.8500; 
  P2G = 0.2935; 
  P3G = 2.4950; 
  PEP = 1.2050; 
  PYR = 1.1000; 
  GLYCERAL3P = 0.0160; 
  ADP = 1.0750; 
  AMP = 0.2750; 
  DHAP = 0.2700; 
  GLYC3P = 0.1035; 
  GLYCEROL = 0.0500; 
  ETOH = 0.0500; 
  G1P = 0.1800; 
  UTP = 0.9200; 
  UDP = 0.3700; 
  UDP_GLC = 1.9250; 
  TRE = 8.1100;   
  T6P = 0.1870; 
  PHOS = 10;
  IMP = 0; 
  INO = 0;
  HYP = 0; 
  ETOHec = 0; 
  GLYCEROLec = 0; 
  GLCec = 0.1830;
  TREec = 0;
  TREvac = 72.9700;
  Vbroth = 3.8940;
  Glycogen_cyt = 100;
  
  
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
  p_GLT_KmGLTGLCi = 0.9041; 
  p_GLT_KmGLTGLCo = 0.9041;
  p_GLT_VmGLT = 1.7021;
  
  p_HXK1_Kadp = 0.3492; #0.4519; # 0.3492 # control here ! 
  p_HXK1_Katp = 0.0931; #0.0804; # 0.0931 # control here ! 
  p_HXK1_Keq = 3.7213e+03 ; #2.0788e+03; # 3.7213e+03 # control here ! 
  p_HXK1_Kg6p = 34.7029; #31.7149; # 34.7029 # control here ! 
  p_HXK1_Kglc = 0.0796;
  p_HXK1_Kt6p = 0.0363;
  p_HXK1_kcat = 19.5886; 
  
  p_PFK_Camp = 0.0287; #0.0240; #0.0287; # control here ! 
  p_PFK_Catp = 1.2822; # 1.9261; #1.2822; # control here ! 
  p_PFK_Cf16bp = 2.3638; #4.0278; #2.3638; # control here ! 
  p_PFK_Cf26bp = 0.0182; #0.0283; # control here ! 
  p_PFK_Ciatp = 40.3824; #40.7309; #40.3824; # control here ! 
  p_PFK_Kamp = 0.0100; # 0.0064; #0.0100; # control here ! 
  p_PFK_Katp = 1.9971; # 1.9911; #1.9971; # control here ! 
  p_PFK_Kf16bp = 0.0437; # 0.0205; #0.0437; # control here ! 
  p_PFK_Kf26bp = 0.0012; # 0.0016; #0.0012; # control here ! 
  p_PFK_Kf6p = 0.0589; # 0.0995; #0.0589; # control here ! 
  p_PFK_Kiatp = 4.9332; # 10.8134; #4.9332; # control here ! 
  p_PFK_L = 1.3886; # 0.7815; #1.3886; # control here ! 
  p_PFK_gR = 1.8127; # 1.6901; #1.8127; # control here ! 
  p_PFK_kcat = 8.7826; # 21.6756; #8.7826; # control here ! 
  p_PFK_F26BP = 1e-3;  # 0.0018; #1e-3;  # control here ! 
  
  p_PGI1_Keq = 0.9564;
  p_PGI1_Kf6p = 7.2433; 
  p_PGI1_Kg6p = 7.9987;
  p_PGI1_kcat = 2.3215; 
  
  p_FBA1_Kdhap = 0.0300;
  p_FBA1_Keq = 0.1223;
  p_FBA1_Kf16bp = 0.6872;
  p_FBA1_Kglyceral3p = 3.5582;
  p_FBA1_kcat = 4.4067;
  
  p_TPI1_Kdhap = 1.2887;
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
  p_PYK1_Katp = 9.3000;  # 93; #9.3000;  # control here ! 
  p_PYK1_Kf16bp = 0.1732;  #0.2; #0.1732;  # control here ! 
  p_PYK1_Kpep = 0.2810;  # 0.5610; #0.2810;  # control here ! 
  p_PYK1_L = 60000;  # 17918; #60000;  # control here ! 
  p_PYK1_hill = 4;
  p_PYK1_kcat = 9.3167;  # 7.4371; #9.3167;  # control here ! 
  
  p_TDH1_Keq = 0.0054;
  p_TDH1_Kglyceral3p = 0.5145;
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
  
  p_HOR2_Kglyc3p = 2.5844;  # 1.5101; #2.5844;  # control here ! 
  p_HOR2_Kpi = 2.5491;
  p_HOR2_kcat = 1.2748;  # 2.2437; #1.2748;  # control here ! 
  
  p_PGM1_Keq = 4.0818;
  p_PGM1_Kg1p = 0.1316;
  p_PGM1_Kg6p = 0.0154;
  p_PGM1_kcat = 4.1018;
  
  p_TPS1_Kg6p = 0.4422;
  p_TPS1_Kudp_glc = 0.1182;
  p_TPS1_kcat = 1.3662e+04;
  p_TPS1_Kpi = 0.2863;
  p_TPS1_KmF6P = 0.7116;
  
  p_TPS2_Kt6p = 0.2427;
  p_TPS2_kcat = 20.7620;
  p_TPS2_Kpi = 0.6991;
  
  p_NTH1_Ktre = 0.1299;
  p_NTH1_kcat = 284.2528;
  
  p_UGP_kcat = 1.4427e+03;
  p_UGP_Kutp = 0.9797;
  p_UGP_Ki_utp = 0.2387;
  p_UGP_Kg1p = 0.1321;
  p_UGP_Ki_udp_glc = 0.0163;
  
  p_ATH1_Ktre = 6.1624e+03;
  p_ATH1_kcat = 546.7721;
  p_AGT1_kcat = 476.4645;
  p_AGT1_Ktre = 0.0855;
  p_AGT1_Keq = 7.3000;
  p_AGT1_Ktre_ec = 0.6846;
  p_AGT1_Ki = 18.0908;
  
  p_ATH1_Kt6p = 0.1000;
  p_ATH1_Ktre_ec = 6.1624e+03;
  p_ATH1_kcat_ec = 546.7721;
  p_ATH1_Kt6p_ec = 0.1000;
  
  p_vacuoleT_vmax = 6.6697e-05;
  p_vacuoleT_Ktre = 2.8274;
  p_vacuoleT_Keq = 1;
  
  p_GlycerolTransport = 0.1001;
  p_kETOHtransport = 0.0328;
  
  p_mitoNADHVmax = 0.2401;
  p_mitoNADHKm = 0.0012;
  p_mitoVmax = 0.7547;
  p_mitoADPKm = 0.3394;
  p_mitoPiKm = 0.4568;
  p_ATPaseK = 0.2219;
  
  p_vacuolePi_k = 0.1699; #0.2619; #0.1699;
  p_vacuolePi_steadyStatePi = 10;
  
  p_ADK1_k = 77.3163;
  p_ADK1_Keq = 0.2676;
  
  km_sinkACE = 1e-04;
  km_sinkF6P = 1e-02; 
  km_sinkG6P = 1e-01;
  km_sinkGAP = 5e-04;
  km_sinkP3G = 1e-03;
  km_sinkPEP = 1e-03;
  km_sinkPYR = 1e-02;
  poly_sinkACE = -0.034836166800000; 
  poly_sinkF6P = 0.024574614000000; 
  poly_sinkG6P =-0.077853600000000; 
  poly_sinkGAP = 0.012626909700000; 
  poly_sinkP3G =-0.007881000000000; 
  poly_sinkPEP =-0.007607000000000; 
  poly_sinkPYR =-0.016132830000000; 
  
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
  f_TPS1 = 0.0014; 
  f_UGP = 0.0024;  
  f_AGT1 = 6.7000e-05;
  f_ATH1 = 0.0020;
  f_ATH1ec = 0.0018;
  f_ATH1vac = 1.9600e-04;
  Csmin = 0.0940; 
  
  # Feast famine cycle-dependent setup
  d = 0.1; # h-1
  GLCin = 7500 / 180; # g/L -> mol/L -> mmol/L
  # Fin = d * Vbroth / 3600; # L/h -> L/s;
  # Fout = Fin;
  # glycSynth_K = 0.0021;
  # glycDeg_K = 5.7535e-04;
  # temp_v_glycSynth = 0;
  # temp_v_glycDeg = 0;
  
  
  #  Fin  := piecewise(43e-3/20, time < 20.1, 0);
  #  Fout := piecewise(0, time < 20.1, 0.166e-3, time < 280.1, 0)
  # sin_y1 = sin(time / 22.5 + 2.68) * 0.0101 + 0.0109;
  # sin_y2 = sin(time / 87 + 4.6) * 0.0101 + 0.0115;
  # sin_y1d = sin(time / 12 + 2.1) * 0.0101 + 0.0109;
  # sin_y2d = sin(time / 110 + 4.6) * 0.0101 + 0.01;
  #   glycSynth_K = 0;
  #   glycDeg_K = 0;
  #   temp_v_glycSynth  := piecewise(sin_y2, time > 61, sin_y1)
  #   temp_v_glycDeg    := piecewise(sin_y2d, time > 31, sin_y1d)

  # Feeding for all the cycles
  
  
  Fin  := piecewise(43e-3/20, time < 20.1, 0, time < 400.1, 43e-3/20, time < 420.1, 0, time < 800.1, 43e-3/20, time < 820.1, 0, time < 1200.1, 43e-3/20, time < 1220.1, 0, time < 1600.1, 43e-3/20, time < 1620.1, 0, time < 2000.1, 0);
  Fout := piecewise(0, time < 20.1, 0.166e-3, time < 280.1, 0, time < 420.1, 0.166e-3, time < 680.1, 0, time < 820.1, 0.166e-3, time < 1080.1, 0, time < 1220.1, 0.166e-3, time < 1480.1, 0, time < 1620.1, 0.166e-3, time < 1880.1, 0)
  sin_y1 = sin(time / 22.5 + 2.68) * 0.0101 + 0.0109;
  sin_y2 = sin(time / 87 + 4.6) * 0.0101 + 0.0115;
  sin_y1d = sin(time / 12 + 2.1) * 0.0101 + 0.0109;
  sin_y2d = sin(time / 110 + 4.6) * 0.0101 + 0.01;
  glycSynth_K = 0;
  glycDeg_K = 0;
  temp_v_glycSynth  := piecewise(sin_y2, time > 61, sin_y1, time > 401, sin_y2, time > 461, sin_y1, time > 801, sin_y2, time > 861, sin_y1, time > 1201, sin_y2, time > 1261, sin_y1, time > 1601, sin_y2, time > 1661, sin_y1, time > 2001, 0);
  temp_v_glycDeg    := piecewise(sin_y2d, time > 31, sin_y1d, time > 401, sin_y2d, time > 431, sin_y1d, time > 801, sin_y2d, time > 831, sin_y1d, time > 1201, sin_y2d, time > 1231, sin_y1d, time > 1601, sin_y2d, time > 1631, sin_y1d, time > 2001, 0);
   
  
end''')

# simulation
# result = r.simulate(0, 2, 3)
# result = r.simulate(0, 20000, 20001)
# result = r.simulate(0, 400, 401)
result = r.simulate(0, 2000, 2001)
r.plot(result)
print(result[-1,:])

# plt.figure()
# temp_id = 33;
# temp_id = 16;
# plt.plot(result[1200:1600,0],result[1200:1600,temp_id],'b-')
# print(result[-1,temp_id])
# plt.show()

# -----------------------------------------------------------------------------

# str_sbml = r.getSBML()
# print(str_sbml)
