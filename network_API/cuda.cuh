#pragma once

#define CUDA_CALCULATION_BLOCK_THREAD_SIZE	256
#define CUDA_CALCULATION_MINIMAL_THREADS 	8192

#include "cuda_runtime.h"

struct CudaParameter
{
public:
	int CountBlock;
	int ThreadPerResult;

	void Set(int countResult, int countCalculationPer);
};

namespace Cuda
{
	// Allocation
	template <typename T>
	T* CUDA_Array_Allocate(int size)
	{
		size_t free_t, total_t;
		cudaMemGetInfo(&free_t, &total_t);

		T* result;
		cudaError cudaStatus = cudaMalloc((void**)&result, size * sizeof(T));
		if (cudaStatus != cudaSuccess)
		{
			int i = 0;
			i++;
		}
		return result;
	}

	void CUDA_Reset();

	void CUDA_Read_CUDA_Data(double * cudaValues, double ** target, int size);

	void CUDA_Read_CUDA_Data(double * cudaValues, double ** target, int size, int offset);

	void CUDA_Calculation_Error();

	int CUDA_Get_Block_Size(int threads);

	// Arrays
	void CUDA_Array_Free(void** address);
	void CUDA_Array_Initialize(double* target, double value, int count);
	bool CUDA_Array_CopyDeviceToHost(void* src, int size, void* dest);
	bool CUDA_Array_CopyHostToDevice(void* src, int size, void* dest);
	bool CUDA_Array_CopyDeviceToDevice(void* src, int size, void* dest);
};