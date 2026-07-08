#include "lg_data.h"
static void sh_exit(void){ register int op asm("r0")=0x18; register int c asm("r1")=0x20026;
  asm volatile("bkpt 0xab"::"r"(op),"r"(c):"memory"); }
static float fa(float x){ return x<0?-x:x; }
static float my_sqrt(float x){ if(x<=0.0f)return 0.0f; float g=x>1.0f?x*0.5f:1.0f;
  for(int k=0;k<8;k++) g=0.5f*(g+x/g); return g; }
static float my_exp(float x){ const float LN2=0.69314718f;
  int n=(int)(x/LN2+(x>=0?0.5f:-0.5f)); float r=x-(float)n*LN2;
  float er=1.0f+r*(1.0f+r*(0.5f+r*(0.16666667f+r*(0.04166667f+r*0.008333333f))));
  float p=1.0f; if(n>=0){for(int k=0;k<n;k++)p*=2.0f;} else {for(int k=0;k<-n;k++)p*=0.5f;} return p*er; }
volatile float sink=0.0f;
static float lg_descriptor(float off){
  float sig=0.0f; volatile const float* VX=(volatile const float*)X; for(int i=0;i<D;i++) sig+=W[i]*VX[i]; sig+=off;
  float eta=ETA_OFF+ETA_SCALE*sig; float hi=SPINODAL*0.999f,lo=0.001f;
  if(eta<lo)eta=lo; if(eta>hi)eta=hi;
  float disc=9.0f*C_PARAM*C_PARAM-32.0f*B_PARAM*eta; float R=0,barrier=0,Gamma=1.0f;
  if(disc>0.0f&&eta>0.0f){ float psi_b=(3.0f*C_PARAM-my_sqrt(disc))/(8.0f*B_PARAM);
    float ab=fa(psi_b); float F=eta*psi_b*psi_b-C_PARAM*ab*ab*ab+B_PARAM*psi_b*psi_b*psi_b*psi_b;
    float dF=F>0.0f?F:0.0f; barrier=dF; R=dF/KBT; }
  Gamma=R>0.0f?my_exp(-R):1.0f; float kappa=2.0f*eta;
  return R+barrier+Gamma+kappa;
}
int main(void){
#if defined(__ARM_FP)
  *((volatile unsigned int*)0xE000ED88u) |= (0xFu<<20); asm volatile("dsb":::"memory"); asm volatile("isb":::"memory");
#endif
  for(int r=0;r<REPS;r++) sink+=lg_descriptor((float)r*1e-9f); sh_exit(); return 0; }
