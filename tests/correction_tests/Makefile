# Makefile for Blackwell SpMM benchmark
CUDA_PATH ?= /usr/local/cuda
NVCC      := $(CUDA_PATH)/bin/nvcc -ccbin g++

INCLUDES  := -I../../build -I../../kernel_benchmark -I../../csrc -I$(CUDA_PATH)/include
LIBS      := -L../../build -lSpMM_API -lcublas -lcusparse -lcuda

SMS       := 120
GENCODE   := $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

NVCCFLAGS := -lineinfo --std=c++11 $(GENCODE)
# NVCCFLAGS += -g -G # for debug

.PHONY: all clean run

all: spmm_test

spmm_test: spmm_test.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f spmm_test

run: spmm_test
	LD_LIBRARY_PATH=../../build:$$LD_LIBRARY_PATH ./spmm_test

# LD_LIBRARY_PATH=../../build:$$LD_LIBRARY_PATH compute-sanitizer --tool memcheck --print-limit 1 ./spmm_test
