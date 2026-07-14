// Concatenate the FP32 kernel exactly as the host does, build with -DDF_LINT
// (df = struct) so every un-converted df arithmetic op is a compile error.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
static std::string rf(const char*p){std::ifstream f(p);std::stringstream ss;ss<<f.rdbuf();return ss.str();}
int main(int argc,char**argv){
  int pidx = argc>1?atoi(argv[1]):0;
  const char* dir="/home/ian/builds/AST/PeriodSearch/period_search_opencl_amd/period_search/";
  const char* files[]={"constants.h","df64.cl","GlobalsCL.h","Intrinsics.cl","swap.cl",
    "blmatrix.cl","curv.cl","Curv2.cl","bright.cl","conv.cl","mrqcof.cl","gauss_errc.cl",
    "mrqmin.cl","test.cl","Start.cl"};
  std::string src;
  for(auto f:files){ src += "\n/*==== "; src += f; src += " ====*/\n"; src += rf((std::string(dir)+f).c_str()); }
  std::ofstream("/tmp/klint.cl")<<src;   // for line mapping
  cl_uint np; clGetPlatformIDs(0,0,&np); std::vector<cl_platform_id> P(np); clGetPlatformIDs(np,P.data(),0);
  cl_device_id d; clGetDeviceIDs(P[pidx],CL_DEVICE_TYPE_GPU,1,&d,0);
  char dn[256]; clGetDeviceInfo(d,CL_DEVICE_NAME,256,dn,0);
  cl_int e; cl_context c=clCreateContext(0,1,&d,0,0,&e);
  const char* cs=src.c_str(); size_t l=src.size();
  cl_program pr=clCreateProgramWithSource(c,1,&cs,&l,&e);
  e=clBuildProgram(pr,1,&d,argc>2?argv[2]:"-DDF_LINT -DPS_FP32_OCL -cl-std=CL1.2 -w",0,0);
  size_t ls; clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,0,0,&ls);
  std::vector<char> lg(ls); clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,ls,lg.data(),0);
  fprintf(stderr,"Device: %s | build %s\n", dn, e==CL_SUCCESS?"OK":"FAILED");
  printf("%s\n", lg.data());
  return e==CL_SUCCESS?0:1;
}
