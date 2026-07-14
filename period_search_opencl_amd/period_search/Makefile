# This should work on Linux.  Modify as needed for other platforms.

# READ ME FIRST:
#
# To keep constinstance of the repository and not be relying to a hardcoded paths please create symbolic links that will be outside the path under
# versioning controll and this way will not brake anyones environment setups.
# From your /home/[YOUR_USERNAME] direcory run:
#	ln -s /[OpenSL_SDK_PATH] AMDSKD
#	ln -s /[your_bpinc_path] boinc_src
# or change those paths acordingly to your setup.
#
# Some materials
# https://stackoverflow.com/questions/57361083/gcc-equivalent-of-gs-gl-gy-oi-md-in-msvc

# DO-NOT-CHANGE!
BOINC_DIR =../../../boinc_src
BOINC_API_DIR = $(BOINC_DIR)/api
BOINC_LIB_DIR = $(BOINC_DIR)/lib
OPENCL_PATH =../../../AMDSDK
# OPENCL_PATH = /usr
OPENCL_LIB_PATH =/opt/amdgpu-pro/lib/x86_64-linux-gnu
OBJ_DIR =build/Release

# END DO-NOT-CHANGE

# Precision toggle (mirrors the CUDA/HIP apps): the default build is FP64
# (native double kernels, results identical to the reference implementations;
# needs cl_khr_fp64/cl_amd_fp64). `make FP32=1` opts into the df64
# double-float build for devices WITHOUT hardware FP64 — see FP32_HANDOFF.md
# for its accuracy envelope. Diagnostic modes passed via CPPFLAGS
# (-DPS_HYBRID, -DPS_DF_DEBUG [FP32+dumps], explicit -DPS_REAL64) suppress
# the default so they keep meaning what they say.
# NOTE: make cannot see flag changes — rm -f build/Release/Start_OpenCl.o
# when switching precision or diagnostic modes.
ifeq ($(FP32),1)
PRECISION_FLAGS =
else ifneq ($(findstring PS_,$(CPPFLAGS)),)
PRECISION_FLAGS =
else
PRECISION_FLAGS = -DPS_REAL64
endif

# CXXFLAGS = -g -O3 .
CXXFLAGS = -O2 -w -s -std=c++17 -m64 -fstack-protector-strong -fstack-clash-protection -ffunction-sections \
	-DNDEBUG \
	-DAMD \
	-DCL_HPP_ENABLE_EXCEPTIONS \
	-I$(BOINC_DIR) \
	-I$(BOINC_LIB_DIR) \
	-I$(BOINC_API_DIR) \
	-I$(OPENCL_PATH)/include \
	-L$(OPENCL_LIB_PATH) \
	-L /usr/X11R6/lib \
	-L. 

CC=g++
CXX=g++
# OPTFLAGS=
OPTFLAGS= -O2 -s -w -m64
CFLAGS=$(OPTFLAGS) \
	-DNDEBUG \
	-DAMD \
	-DCL_HPP_ENABLE_EXCEPTIONS \
	-I$(BOINC_DIR) \
	-I$(BOINC_LIB_DIR) \
	-I$(BOINC_API_DIR) \
	-I$(OPENCL_PATH)/include \
	-L$(OPENCL_LIB_PATH) \
	-L. 

LDFLAGS=-lm -lrt -Xlinker -rpath -m64 .

MY_LIBS=$(OBJ_DIR)/trifac.o $(OBJ_DIR)/areanorm.o $(OBJ_DIR)/sphfunc.o $(OBJ_DIR)/ellfit.o $(OBJ_DIR)/ludcmp.o $(OBJ_DIR)/lubksb.o\
	$(OBJ_DIR)/covsrt.o $(OBJ_DIR)/memory.o $(OBJ_DIR)/host_dot_product.o\
	$(OBJ_DIR)/compareResult.o $(OBJ_DIR)/Start_OpenCl.o $(OBJ_DIR)/VersionInfo.o 

SOURCE_FILES = trifac.cpp areanorm.cpp sphfunc.cpp ellfit.cpp ludcmp.cpp lubksb.cpp\
	covsrt.cpp memory.cpp host_dot_product.cpp\
	compareResult.cpp Start_OpenCl.cpp VersionInfo.cpp

OBJECT_FILES     = $(SOURCE_FILES:%.cpp=$(OBJ_DIR)/%.o)

PROGS = period_search_BOINC_linux_amd_opencl_110_x64_Release
EXECUTABLE_FILES = $(PROGS:%=$(OBJ_DIR)/%)

all: $(EXECUTABLE_FILES)

# The OpenCL kernels the app really runs are the ocl_src_kernelSource string in
# kernels.cpp (the .cl files are only read in _DEBUG builds). Regenerate it
# whenever any kernel source changes; order must match the runtime concatenation
# in Start_OpenCl.cpp.
KERNEL_SRCS = constants.h df64.cl GlobalsCL.h Intrinsics.cl swap.cl blmatrix.cl\
	curv.cl Curv2.cl bright.cl conv.cl mrqcof.cl gauss_errc.cl mrqmin.cl\
	test.cl Start.cl

kernels.cpp: $(KERNEL_SRCS) gen_kernels_cpp.py
	python3 gen_kernels_cpp.py $@ $(KERNEL_SRCS)

$(OBJ_DIR)/Start_OpenCl.o: kernels.cpp

$(OBJ_DIR)/libstdc++.a:
	ln -s `g++ -print-file-name=libstdc++.a` $(OBJ_DIR)/libstdc++.a

$(OBJ_DIR)/libOpenCL.so:
	ln -s $(OPENCL_LIB_PATH)/libOpenCL.so $(OBJ_DIR)/libOpenCL.so

clean:
	/bin/rm -f $(OBJ_DIR)/$(PROGS) *.o $(OBJ_DIR)/*.o $(OBJ_DIR)/libstdc++.a $(OBJ_DIR)/libOpenCL.so

distclean:
	/bin/rm -f $(OBJ_DIR)/$(PROGS) $(OBJ_DIR)/*.o $(OBJ_DIR)/libstdc++.a $(OBJ_DIR)/libOpenCL.so

install: period_search_BOINC_linux_amd_opencl_110_x64_Release

# specify library paths explicitly (rather than -l)
# because otherwise you might get a version in /usr/lib etc.

$(EXECUTABLE_FILES): period_search_BOINC.cpp  $(OBJECT_FILES) $(OBJ_DIR)/libstdc++.a $(BOINC_API_DIR)/libboinc_api.a $(BOINC_LIB_DIR)/libboinc.a $(OBJ_DIR)/libOpenCL.so
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECT_FILES) $< $(OBJ_DIR)/libstdc++.a -pthread \
	$(BOINC_API_DIR)/libboinc_api.a $(OBJ_DIR)/libOpenCL.so \
	$(BOINC_LIB_DIR)/libboinc.a $(LDFLAGS)
	@echo "Build successful!"

$(OBJ_DIR)/%.o: %.cpp
	@echo Compiling $<
	@mkdir -p $(dir $@)
	$(CXX) $(PRECISION_FLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
