import numpy as np, json
from engines import LGPhysicsEngine
z=np.load('/tmp/cmapss_cache/FD003.npz')
Xbt=z['Xbt'].astype(np.float64); ttf_tr=z['ttf_tr'].astype(np.float64)
Xbe=z['Xbe'].astype(np.float64)
eng=LGPhysicsEngine()          # defaults: b=1.0,c=1.5,kB_T_eff=0.5,eta_h=0.85,eta_f=0.05
eng.calibrate(Xbt, ttf_tr)
# escolher uma amostra de teste "interessante" (a meio da vida)
eta_all=eng.get_eta(Xbe); i=int(np.argmax(eta_all))
x=Xbe[i]
ref=eng.transform(x[None,:])[0]   # [R, barrier, Gamma, kappa]
d=len(eng.weights)
print("d =",d)
print("params: b=%.6f c=%.6f kBTeff=%.6f spinodal=%.6f"%(eng.b,eng.c,eng.kB_T_eff,eng.spinodal))
print("eta_offset=%.8f eta_scale=%.8f"%(eng.eta_offset,eng.eta_scale))
print("REFERENCIA Python: R=%.6f barrier=%.6f Gamma=%.6f kappa=%.6f"%(ref[0],ref[1],ref[2],ref[3]))
# escrever header C
with open('/tmp/lg_data.h','w') as f:
    f.write("// auto-gerado a partir do motor LG (FD003)\n#ifndef LG_DATA_H\n#define LG_DATA_H\n")
    f.write("#define D %d\n"%d)
    f.write("static const float B_PARAM=%.8ff;\n"%eng.b)
    f.write("static const float C_PARAM=%.8ff;\n"%eng.c)
    f.write("static const float KBT=%.8ff;\n"%eng.kB_T_eff)
    f.write("static const float SPINODAL=%.8ff;\n"%eng.spinodal)
    f.write("static const float ETA_OFF=%.8ff;\n"%eng.eta_offset)
    f.write("static const float ETA_SCALE=%.8ff;\n"%eng.eta_scale)
    f.write("static const float W[D]={%s};\n"%",".join("%.8ff"%w for w in eng.weights))
    f.write("static const float X[D]={%s};\n"%",".join("%.8ff"%v for v in x))
    f.write("// referencia Python (x1e6): R=%d barrier=%d Gamma=%d kappa=%d\n"%(
        round(ref[0]*1e6),round(ref[1]*1e6),round(ref[2]*1e6),round(ref[3]*1e6)))
    f.write("#endif\n")
json.dump({"R":ref[0],"barrier":ref[1],"Gamma":ref[2],"kappa":ref[3]},open('/tmp/lg_ref.json','w'))
print("header escrito em /tmp/lg_data.h")
