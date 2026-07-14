// Differential test: df64.cl DF_TRIPLE (triple-float, FP32-only) ops
// on-device vs long-double reference on host.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

struct tf { float x, y, z; };

static std::string readfile(const char* p){ std::ifstream f(p); std::stringstream ss; ss<<f.rdbuf(); return ss.str(); }

int main(int argc, char** argv){
    int plat_idx = argc>1?atoi(argv[1]):0;
    cl_uint np=0; clGetPlatformIDs(0,0,&np);
    std::vector<cl_platform_id> plats(np); clGetPlatformIDs(np,plats.data(),0);
    if(plat_idx>=(int)np){ printf("no platform %d\n",plat_idx); return 1; }
    char pn[256]; clGetPlatformInfo(plats[plat_idx],CL_PLATFORM_NAME,256,pn,0);
    cl_device_id dev; cl_uint nd=0;
    if(clGetDeviceIDs(plats[plat_idx],CL_DEVICE_TYPE_GPU,1,&dev,&nd)!=CL_SUCCESS||nd==0){
        clGetDeviceIDs(plats[plat_idx],CL_DEVICE_TYPE_ALL,1,&dev,&nd);
    }
    char dn[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,256,dn,0);
    printf("Platform: %s | Device: %s\n", pn, dn);

    cl_int err;
    cl_context ctx = clCreateContext(0,1,&dev,0,0,&err);
    cl_command_queue q = clCreateCommandQueue(ctx,dev,0,&err);

    std::string src = readfile("/home/ian/builds/AST/PeriodSearch/period_search_opencl_amd/period_search/df64.cl");
    src += R"CLC(
typedef struct { float x, y, z; } tf3;
__kernel void test_ops(__global tf3* a,__global tf3* b,__global tf3* ang,
                       __global tf3* sm,__global tf3* xac,__global tf3* out){
  int i=get_global_id(0);
  df A = (df){a[i].x, a[i].y, a[i].z};
  df B = (df){b[i].x, b[i].y, b[i].z};
  df ANG = (df){ang[i].x, ang[i].y, ang[i].z};
  df SM = (df){sm[i].x, sm[i].y, sm[i].z};
  df XA = (df){xac[i].x, xac[i].y, xac[i].z};
  df r[12];
  r[0]=df_add(A,B);
  r[1]=df_sub(A,B);
  r[2]=df_mul(A,B);
  r[3]=df_div(A,B);
  r[4]=df_sqrt(df_fabs(A));
  df s,c; df_sincos(ANG,&s,&c);
  r[5]=s; r[6]=c;
  r[7]=df_exp(SM);
  r[8]=df_log(df_fabs(A));
  r[9]=df_acos(XA);
  /* fmod at app-realistic magnitude (n up to ~1.6e5 << 2^24; the app's
     rotation phase peaks around 2e5) */
  df F = df_mul(ANG, (df){33333.0f, 0.0f, 0.0f});
  r[10]=df_fmod(F, DF_2PI);
  r[11]=df_rcp(B);
  for(int j=0;j<12;j++){ out[i*12+j].x=r[j].x; out[i*12+j].y=r[j].y; out[i*12+j].z=r[j].z; }
}
)CLC";
    const char* csrc = src.c_str(); size_t clen = src.size();
    cl_program prog = clCreateProgramWithSource(ctx,1,&csrc,&clen,&err);
    err = clBuildProgram(prog,1,&dev,"-w -D DF_TRIPLE -cl-std=CL1.2",0,0);
    if(err!=CL_SUCCESS){
        size_t ls; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,0,&ls);
        std::vector<char> log(ls); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,ls,log.data(),0);
        printf("BUILD FAILED:\n%s\n", log.data()); return 2;
    }
    cl_kernel k = clCreateKernel(prog,"test_ops",&err);

    const int N=200000;
    std::vector<tf> a(N),b(N),ang(N),sm(N),xac(N),out(N*12);
    unsigned long long seed=12345;
    auto rnd=[&](){ seed=seed*6364136223846793005ULL+1442695040888963407ULL; return (double)(seed>>11)/(double)(1ULL<<53); };
    auto split=[](long double d,tf&o){
        float h=(float)d; float m=(float)(d-(long double)h); float l=(float)(d-(long double)h-(long double)m);
        o.x=h; o.y=m; o.z=l; };
    std::vector<long double> ad(N),bd(N),angd(N),smd(N),xacd(N);
    for(int i=0;i<N;i++){
        double mag=pow(10.0,rnd()*12-6);
        ad[i]=(long double)((rnd()-0.5)*mag); bd[i]=(long double)((rnd()-0.5)*mag+1e-30);
        if(i%3==1) bd[i]=-ad[i]*(long double)(1.0+(rnd()-0.5)*1e-5); // cancellation
        angd[i]=(long double)(rnd()*2*M_PI);
        smd[i]=(long double)((rnd()-0.5)*40);
        double xc=fmod(fabs((double)ad[i]),2.0)-1.0; xacd[i]=(long double)xc;
        split(ad[i],a[i]); split(bd[i],b[i]); split(angd[i],ang[i]); split(smd[i],sm[i]); split(xacd[i],xac[i]);
    }
    // hard opposition cases for acos
    xacd[0]=1.0L-9e-8L; split(xacd[0],xac[0]);
    xacd[1]=1.0L; split(xacd[1],xac[1]);
    xacd[2]=-1.0L+9e-8L; split(xacd[2],xac[2]);
    xacd[3]=0.49999999L; split(xacd[3],xac[3]);
    xacd[4]=-0.49999999L; split(xacd[4],xac[4]);

    auto buf=[&](std::vector<tf>&v){ return clCreateBuffer(ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,v.size()*sizeof(tf),v.data(),0); };
    cl_mem ba=buf(a),bb=buf(b),bang=buf(ang),bsm=buf(sm),bxac=buf(xac),bout=buf(out);
    clSetKernelArg(k,0,sizeof(cl_mem),&ba); clSetKernelArg(k,1,sizeof(cl_mem),&bb);
    clSetKernelArg(k,2,sizeof(cl_mem),&bang); clSetKernelArg(k,3,sizeof(cl_mem),&bsm);
    clSetKernelArg(k,4,sizeof(cl_mem),&bxac); clSetKernelArg(k,5,sizeof(cl_mem),&bout);
    size_t gs=N; clEnqueueNDRangeKernel(q,k,1,0,&gs,0,0,0,0);
    clEnqueueReadBuffer(q,bout,CL_TRUE,0,out.size()*sizeof(tf),out.data(),0,0,0);

    /* the modulus the device actually uses: the 3-float split of double(2*pi) */
    const long double PI2 = (long double)6.28318548f + (long double)-1.748455531e-07f
                          + (long double)-7.105427358e-15f;
    const char* nm[12]={"add","sub","mul","div","sqrt","sin","cos","exp","log","acos","fmod","rcp"};
    double worst[12]={0}; int wi[12]={0};
    for(int i=0;i<N;i++){
        long double la=(long double)a[i].x+(long double)a[i].y+(long double)a[i].z;
        long double lb=(long double)b[i].x+(long double)b[i].y+(long double)b[i].z;
        long double lang=(long double)ang[i].x+(long double)ang[i].y+(long double)ang[i].z;
        long double lsm=(long double)sm[i].x+(long double)sm[i].y+(long double)sm[i].z;
        long double lxac=(long double)xac[i].x+(long double)xac[i].y+(long double)xac[i].z;
        long double ref[12]={la+lb,la-lb,la*lb,la/lb,sqrtl(fabsl(la)),
            sinl(lang),cosl(lang),expl(lsm),logl(fabsl(la)),acosl(lxac),
            fmodl(lang*33333.0L,PI2), 1.0L/lb};
        for(int j=0;j<12;j++){
            long double got=(long double)out[i*12+j].x+(long double)out[i*12+j].y+(long double)out[i*12+j].z;
            long double den=fabsl(ref[j])>1e-300L?fabsl(ref[j]):1e-300L;
            // sin/cos/log absolute; fmod absolute (the quotient-boundary case
            // can legitimately differ by one period)
            double e;
            if(j==5||j==6||j==8) e=(double)fabsl(got-ref[j]);
            else if(j==10){ long double d=fabsl(got-ref[j]); if(d>PI2/2) d=fabsl(d-PI2); e=(double)d; }
            else e=(double)(fabsl(got-ref[j])/den);
            if(e>worst[j]&&std::isfinite(e)&&fabsl(ref[j])<1e30L){worst[j]=e;wi[j]=i;}
        }
    }
    for(int j=0;j<12;j++)
        printf("%-5s worst %s err = %.3e (2^%.1f)\n", nm[j], (j==5||j==6||j==8||j==10)?"abs":"rel",
               worst[j], worst[j]>0?log2(worst[j]):-999.0);
    return 0;
}
