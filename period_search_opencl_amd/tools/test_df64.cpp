// Differential test: df64.cl ops on-device vs long-double reference on host.
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

static std::string readfile(const char* p){ std::ifstream f(p); std::stringstream ss; ss<<f.rdbuf(); return ss.str(); }

int main(int argc, char** argv){
    int plat_idx = argc>1?atoi(argv[1]):0;
    // gather platforms/devices
    cl_uint np=0; clGetPlatformIDs(0,0,&np);
    std::vector<cl_platform_id> plats(np); clGetPlatformIDs(np,plats.data(),0);
    if(plat_idx>=(int)np){ printf("no platform %d\n",plat_idx); return 1; }
    char pn[256]; clGetPlatformInfo(plats[plat_idx],CL_PLATFORM_NAME,256,pn,0);
    cl_device_id dev; cl_uint nd=0;
    if(clGetDeviceIDs(plats[plat_idx],CL_DEVICE_TYPE_GPU,1,&dev,&nd)!=CL_SUCCESS||nd==0){
        clGetDeviceIDs(plats[plat_idx],CL_DEVICE_TYPE_ALL,1,&dev,&nd);
    }
    char dn[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,256,dn,0);
    char ext[4096]; clGetDeviceInfo(dev,CL_DEVICE_EXTENSIONS,4096,ext,0);
    bool has_fp64 = strstr(ext,"cl_khr_fp64")!=nullptr;
    printf("Platform: %s | Device: %s | cl_khr_fp64: %s\n", pn, dn, has_fp64?"yes":"NO");

    cl_int err;
    cl_context ctx = clCreateContext(0,1,&dev,0,0,&err);
    cl_command_queue q = clCreateCommandQueue(ctx,dev,0,&err);

    std::string src = readfile("/home/ian/builds/AST/PeriodSearch/period_search_opencl_amd/period_search/df64.cl");
    src += R"CLC(
__kernel void test_ops(__global float2* a,__global float2* b,__global float2* ang,
                       __global float2* sm,__global float2* xac,__global float2* out){
  int i=get_global_id(0);
  out[i*10+0]=df_add(a[i],b[i]);
  out[i*10+1]=df_sub(a[i],b[i]);
  out[i*10+2]=df_mul(a[i],b[i]);
  out[i*10+3]=df_div(a[i],b[i]);
  out[i*10+4]=df_sqrt(df_fabs(a[i]));
  df s,c; df_sincos(ang[i],&s,&c);
  out[i*10+5]=s; out[i*10+6]=c;
  out[i*10+7]=df_exp(sm[i]);
  out[i*10+8]=df_log(df_fabs(a[i]));
  out[i*10+9]=df_acos(xac[i]);
}
)CLC";
    const char* csrc = src.c_str(); size_t clen = src.size();
    cl_program prog = clCreateProgramWithSource(ctx,1,&csrc,&clen,&err);
    err = clBuildProgram(prog,1,&dev,"-cl-std=CL1.2",0,0);
    if(err!=CL_SUCCESS){
        size_t ls; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,0,&ls);
        std::vector<char> log(ls); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,ls,log.data(),0);
        printf("BUILD FAILED:\n%s\n", log.data()); return 2;
    }
    cl_kernel k = clCreateKernel(prog,"test_ops",&err);

    const int N=200000;
    std::vector<cl_float2> a(N),b(N),ang(N),sm(N),xac(N),out(N*10);
    std::vector<double> ad(N),bd(N),angd(N),smd(N),xacd(N);
    unsigned long long seed=12345;
    auto rnd=[&](){ seed=seed*6364136223846793005ULL+1442695040888963407ULL; return (double)(seed>>11)/(double)(1ULL<<53); };
    auto split=[](double d,cl_float2&o){ float h=(float)d; o.s[0]=h; o.s[1]=(float)(d-(double)h); };
    for(int i=0;i<N;i++){
        double mag=pow(10.0,rnd()*12-6);
        ad[i]=(rnd()-0.5)*mag; bd[i]=(rnd()-0.5)*mag+1e-30;
        if(i%3==1) bd[i]=-ad[i]*(1.0+(rnd()-0.5)*1e-5); // cancellation
        angd[i]=rnd()*2*M_PI;            // sincos arg in [0,2pi]
        smd[i]=(rnd()-0.5)*40;           // exp arg
        double xc=fmod(fabs(ad[i]),2.0)-1.0; xacd[i]=xc; // acos arg [-1,1]
        split(ad[i],a[i]); split(bd[i],b[i]); split(angd[i],ang[i]); split(smd[i],sm[i]); split(xacd[i],xac[i]);
    }
    // hard opposition cases for acos
    xacd[0]=1.0-9e-8; split(xacd[0],xac[0]);
    xacd[1]=1.0; split(xacd[1],xac[1]);
    xacd[2]=-1.0+9e-8; split(xacd[2],xac[2]);

    auto buf=[&](std::vector<cl_float2>&v){ return clCreateBuffer(ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,v.size()*sizeof(cl_float2),v.data(),0); };
    cl_mem ba=buf(a),bb=buf(b),bang=buf(ang),bsm=buf(sm),bxac=buf(xac),bout=buf(out);
    clSetKernelArg(k,0,sizeof(cl_mem),&ba); clSetKernelArg(k,1,sizeof(cl_mem),&bb);
    clSetKernelArg(k,2,sizeof(cl_mem),&bang); clSetKernelArg(k,3,sizeof(cl_mem),&bsm);
    clSetKernelArg(k,4,sizeof(cl_mem),&bxac); clSetKernelArg(k,5,sizeof(cl_mem),&bout);
    size_t gs=N; clEnqueueNDRangeKernel(q,k,1,0,&gs,0,0,0,0);
    clEnqueueReadBuffer(q,bout,CL_TRUE,0,out.size()*sizeof(cl_float2),out.data(),0,0,0);

    const char* nm[10]={"add","sub","mul","div","sqrt","sin","cos","exp","log","acos"};
    double worst[10]={0}; int wi[10]={0};
    for(int i=0;i<N;i++){
        // reference built from the df64-ROUNDED inputs (what the kernel actually sees)
        long double la=(long double)a[i].s[0]+(long double)a[i].s[1];
        long double lb=(long double)b[i].s[0]+(long double)b[i].s[1];
        long double lang=(long double)ang[i].s[0]+(long double)ang[i].s[1];
        long double lsm=(long double)sm[i].s[0]+(long double)sm[i].s[1];
        long double lxac=(long double)xac[i].s[0]+(long double)xac[i].s[1];
        long double ref[10]={la+lb,la-lb,la*lb,la/lb,sqrtl(fabsl(la)),
            sinl(lang),cosl(lang),expl(lsm),logl(fabsl(la)),acosl(lxac)};
        for(int j=0;j<10;j++){
            long double got=(long double)out[i*10+j].s[0]+(long double)out[i*10+j].s[1];
            long double den=fabsl(ref[j])>1e-300L?fabsl(ref[j]):1e-300L;
            double e = (j==5||j==6)? (double)fabsl(got-ref[j]) : (double)(fabsl(got-ref[j])/den);
            if(e>worst[j]&&std::isfinite(e)){worst[j]=e;wi[j]=i;}
        }
    }
    for(int j=0;j<10;j++)
        printf("%-5s worst %s err = %.3e (2^%.1f)\n", nm[j], (j==5||j==6)?"abs":"rel",
               worst[j], worst[j]>0?log2(worst[j]):-999.0);
    return 0;
}
