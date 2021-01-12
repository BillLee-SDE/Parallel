#include "cuda.cuh"
#include "device_launch_parameters.h"

void CudaParameter::Set(int countResult, int countCalculationPer)
{
	int countThread = 0;

	if (countResult >= CUDA_CALCULATION_MINIMAL_THREADS)
	{
		ThreadPerResult = 1;
		countThread = countResult;
	}
	else
	{
		int temp = 1;

		do
		{
			ThreadPerResult = temp;
			temp = ThreadPerResult * 2;
		} while (temp <= countCalculationPer && temp <= CUDA_CALCULATION_BLOCK_THREAD_SIZE);

		countThread = ThreadPerResult * countResult;
	}

	CountBlock = (countThread + CUDA_CALCULATION_BLOCK_THREAD_SIZE - 1) / CUDA_CALCULATION_BLOCK_THREAD_SIZE;
}


namespace Cuda
{
	void CUDA_Calculation_Error()
	{
		cudaError cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			int i = 0;
			i++;
		}
	}


	void CUDA_Reset()
	{
		if (cudaDeviceReset() != cudaSuccess) { CUDA_Calculation_Error(); }
	}

	void CUDA_Read_CUDA_Data(double* cudaValues, double** target, int size)
	{
		if (*target != nullptr) { delete *target; }
		*target = new double[size];
		CUDA_Array_CopyDeviceToHost(cudaValues, size * sizeof(double), *target);
	}

	void CUDA_Read_CUDA_Data(double* cudaValues, double** target, int size, int offset)
	{
		if (*target != nullptr) { delete *target; }
		*target = new double[size];
		CUDA_Array_CopyDeviceToHost(&cudaValues[offset], size * sizeof(double), *target);
	}

	int CUDA_Get_Block_Size(int threads)
	{
		return (threads + CUDA_CALCULATION_BLOCK_THREAD_SIZE - 1) / CUDA_CALCULATION_BLOCK_THREAD_SIZE;
	}

	void CUDA_Array_Free(void ** address)
	{
		if (*address)
		{
			cudaFree(*address);
		}
		*address = nullptr;
	}

	__global__ void cuCalulation_kernel_Initialize(double * target, double value, int count) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < count)
		{
			target[index] = value;
		}
	}

	void CUDA_Array_Initialize(double * target, double value, int count)
	{
		cuCalulation_kernel_Initialize << <CUDA_Get_Block_Size(count), CUDA_CALCULATION_BLOCK_THREAD_SIZE >> > (target, value, count);
		CUDA_Calculation_Error();
	}

	bool CUDA_Array_CopyDeviceToHost(void* src, int size, void* dest)
	{
		cudaError cudaStatus = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			int i = 0;
			i++;
		}
		return true;
	}

	bool CUDA_Array_CopyHostToDevice(void* src, int size, void* dest)
	{
		cudaError cudaStatus = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			int i = 0;
			i++;
		}
		return true;
	}

	bool CUDA_Array_CopyDeviceToDevice(void* src, int size, void* dest)
	{
		cudaError cudaStatus = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess)
		{
			int i = 0;
			i++;
		}
		return true;
	}
}