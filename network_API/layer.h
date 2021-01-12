#pragma once

#include "cuda.cuh"
#include "global.h"

class Layer
{
public:
	double* Output();
	double* Signal();
	
	virtual int Size();
	virtual int MaxBatchSize();

	virtual int ConvRowCount();
	virtual int ConvRowSize();
	virtual int ConvFilter();
	
	virtual void DropoutSelect();

	virtual void Forward(int sets, bool training);
	virtual void Backward(int sets, int* dropoutSelect);
	
protected:
	Layer* _source;

	double* _output;
	double* _signal;
};

class Input : public Layer
{
public:
	Input(int iRow, int iSize, int iBatch);
	~Input();

	int Size();
	int MaxBatchSize();

	int ConvRowCount();
	int ConvRowSize();
	int ConvFilter();

	void DropoutSelect();

	void Forward(int sets, bool training);
	void Backward(int sets, int* dropoutSelect);

private:
	int _rowCount;
	int _rowSize;
	int _maxBatchSize;
};

class CalculationC_1D : public Layer
{
public:
	CalculationC_1D(Layer* source, int window, bool bias, Global::ActivationFunction activationType, int filter);
	~CalculationC_1D();

	int Size();

	int ConvRowCount();
	int ConvRowSize();
	int ConvFilter();

	void DropoutSelect();

	void Forward(int sets, bool training);
	void Backward(int sets, int* dropout);

private:
	double* _weight[5];
	double* _bias[5];

	int _window;
	Global::ActivationFunction _activationType;

	int _convFilter;
	int _convRowCount;
	int _convRowSize;

	CudaParameter _parameterBias;
	CudaParameter _parameterWeight;
};

class CalculationC_2D : public Layer
{
public:
	CalculationC_2D(Layer* source, int windowHeight, int windowWidth, bool bias, Global::ActivationFunction activationType, int filter);
	~CalculationC_2D();

	int Size();

	int ConvRowCount();
	int ConvRowSize();
	int ConvFilter();

	void DropoutSelect();

	void Forward(int sets, bool training);
	void Backward(int sets, int* dropout);

private:
	double* _weight[5];
	double* _bias[5];

	int _windowHeight;
	int _windowWidth;
	Global::ActivationFunction _activationType;

	int _convFilter;
	int _convHeight;
	int _convWidth;

	CudaParameter _parameterBias;
	CudaParameter _parameterWeight;
};

class Pooling_1D : public Layer
{
public:
	Pooling_1D(Layer* source, int pooling, Global::PoolingType poolingType);
	~Pooling_1D();

	int Size();

	int ConvRowCount();
	int ConvRowSize();
	int ConvFilter();

	void DropoutSelect();

	void Forward(int sets, bool training);
	void Backward(int sets, int* dropoutSelect);

private:
	double* _pooling;	

	int _sizePooling;
	Global::PoolingType _poolingType;
	int _convRowSize;
};

class Pooling_2D : public Layer
{
public:
	Pooling_2D(Layer* source, int poolingHeight, int poolingWidth, Global::PoolingType poolingType);
	~Pooling_2D();

	int Size();

	int ConvRowCount();
	int ConvRowSize();
	int ConvFilter();

	void DropoutSelect();

	void Forward(int sets, bool training);
	void Backward(int sets, int* dropoutSelect);

private:
	double* _pooling;

	int _poolingHeight;
	int _poolingWidth;
	Global::PoolingType _poolingType;
	int _convHeight;
	int _convWidth;
};

class CalculationFC : public Layer
{
public:
	CalculationFC(Layer* source, bool bias, double dropoutActive, Global::ActivationFunction activationType, int oSize);
	~CalculationFC();

	int Size();

	int ConvRowCount();
	int ConvRowSize();
	int ConvFilter();

	void DropoutSelect();

	void Forward(int sets, bool training);
	void Backward(int sets, int* dropoutSelect);

private:
	double _dropoutScale;
	int* _dropoutTrack;
	int _dropoutCharges;
	int _dropoutSelectCount;
	int* _dropoutSelectIndex;

	double* _weight[5];
	double* _bias[5];

	Global::ActivationFunction _activationType;

	int _size;

	CudaParameter _parameterSignal;
	CudaParameter _parameterBias;
	CudaParameter _parameterWeight;
	CudaParameter _parameterForward;
};
