#include "layer.h"
#include "calculation.cuh"
#include <random>
#include <math.h>

using namespace std;

double * Layer::Output()
{
	return _output;
}

double * Layer::Signal()
{
	return _signal;
}

int Layer::Size()
{
	return 0;
}

int Layer::MaxBatchSize()
{
	return _source->MaxBatchSize();
}

int Layer::ConvRowCount()
{
	return 0;
}

int Layer::ConvRowSize()
{
	return 0;
}

int Layer::ConvFilter()
{
	return 0;
}

void Layer::DropoutSelect() { }

void Layer::Forward(int sets, bool training) { }

void Layer::Backward(int sets, int * dropoutSelect) { }


Input::Input(int iRow, int iSize, int iBatch)
{
	_rowCount = iRow;
	_rowSize = iSize;
	_maxBatchSize = iBatch;
	_source = nullptr;
	_output = Cuda::CUDA_Array_Allocate<double>(_rowCount * _rowSize * _maxBatchSize);
	_signal = nullptr;
}

Input::~Input()
{
	Cuda::CUDA_Array_Free((void**)&_output);
}

int Input::Size()
{
	return _rowCount * _rowSize;
}

int Input::MaxBatchSize()
{
	return _maxBatchSize;
}

int Input::ConvRowCount()
{
	return _rowCount;
}

int Input::ConvRowSize()
{
	return _rowSize;
}

int Input::ConvFilter()
{
	return 1;
}

void Input::DropoutSelect() { }

void Input::Forward(int sets, bool training) {}

void Input::Backward(int sets, int * dropoutSelect) {}


CalculationC_1D::CalculationC_1D(Layer * source, int window, bool bias, Global::ActivationFunction activationType, int filter)
{
	_source = source;
	_window = window;
	_activationType = activationType;
	_convFilter = filter;
	_convRowCount = _source->ConvRowCount();
	_convRowSize = _source->ConvRowSize() - _window + 1;

	_output = Cuda::CUDA_Array_Allocate<double>(_convRowSize * _convRowCount * _convFilter * MaxBatchSize());
	_signal = Cuda::CUDA_Array_Allocate<double>(_convRowSize * _convRowCount * _convFilter * MaxBatchSize());

	int sizeWeight = _convFilter * _source->ConvFilter() * _convRowCount * _window;
	_weight[0] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[1] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[2] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[3] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[4] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);

	Cuda::CUDA_Array_Initialize(_weight[1], 1.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[2], 1.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[3], 0.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[4], 0.0, sizeWeight);

	default_random_engine generator(random_device{}());
	mt19937 e2(generator());
	normal_distribution<double> distribution(0.0, 1.0);
	double* temp = new double[sizeWeight];
	double randomSD = 1.0 / sqrt(_source->ConvFilter() * _window);
	for (int i = 0; i < sizeWeight; i++) { temp[i] = distribution(generator) * randomSD; }
	Cuda::CUDA_Array_CopyHostToDevice(temp, sizeWeight * sizeof(double), _weight[0]);
	delete temp;

	_parameterWeight.Set(sizeWeight, MaxBatchSize() * _convRowSize);

	if (bias)
	{
		int sizeBias = _convFilter * _convRowCount;
		_bias[0] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[1] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[2] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[3] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[4] = Cuda::CUDA_Array_Allocate<double>(sizeBias);

		Cuda::CUDA_Array_Initialize(_bias[0], 0.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[1], 1.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[2], 1.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[3], 0.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[4], 0.0, sizeBias);
		
		_parameterBias.Set(sizeBias, MaxBatchSize() * _convRowSize);
	}
}

CalculationC_1D::~CalculationC_1D()
{
	Cuda::CUDA_Array_Free((void**)&_output);
	Cuda::CUDA_Array_Free((void**)&_signal);

	for (int i = 0; i < 5; i++)
	{
		Cuda::CUDA_Array_Free((void**)&_weight[i]);
		Cuda::CUDA_Array_Free((void**)&_bias[i]);
	}

	delete _source;
}

int CalculationC_1D::Size()
{
	return _convFilter * _convRowCount * _convRowSize;
}

int CalculationC_1D::ConvRowCount()
{
	return _convRowCount;
}

int CalculationC_1D::ConvRowSize()
{
	return _convRowSize;
}

int CalculationC_1D::ConvFilter()
{
	return _convFilter;
}

void CalculationC_1D::DropoutSelect() {}

void CalculationC_1D::Forward(int sets, bool training)
{
	_source->Forward(sets, training);

	Calculation::Calculation_Convolution_1D_Forward(
		sets, _convRowCount, _source->ConvFilter(), _source->ConvRowSize(),
		_source->Output(), _window, _weight[0], _bias[0], _activationType,
		_convFilter, _convRowSize, _output);
}

void CalculationC_1D::Backward(int sets, int* dropoutSelect)
{
	Calculation::Calculation_Convolution_1D_Backward(
		sets, _convRowCount, _source->ConvFilter(), _source->ConvRowSize(),
		_source->Signal(), _source->Output(), _window, _weight, _bias,
		_activationType, _convFilter, _convRowSize, _output, _signal,
		_parameterBias, _parameterWeight);	
		
	_source->Backward(sets, dropoutSelect);
}


CalculationC_2D::CalculationC_2D(Layer* source, int windowHeight, int windowWidth, bool bias, Global::ActivationFunction activationType, int filter)
{
	_source = source;
	_windowHeight = windowHeight;
	_windowWidth = windowWidth;
	_activationType = activationType;
	_convFilter = filter;
	_convHeight = _source->ConvRowCount() - _windowHeight + 1;
	_convWidth = _source->ConvRowSize() - _windowWidth + 1;

	_output = Cuda::CUDA_Array_Allocate<double>(_convHeight * _convWidth * _convFilter * MaxBatchSize());
	_signal = Cuda::CUDA_Array_Allocate<double>(_convHeight * _convWidth * _convFilter * MaxBatchSize());

	int sizeWeight = _convFilter * _source->ConvFilter() * _windowHeight * _windowWidth;
	_weight[0] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[1] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[2] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[3] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[4] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);

	Cuda::CUDA_Array_Initialize(_weight[1], 1.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[2], 1.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[3], 0.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[4], 0.0, sizeWeight);

	default_random_engine generator(random_device{}());
	mt19937 e2(generator());
	normal_distribution<double> distribution(0.0, 1.0);
	double* temp = new double[sizeWeight];
	double randomSD = 1.0 / sqrt(_source->ConvFilter() * _windowHeight * _windowWidth);
	for (int i = 0; i < sizeWeight; i++) { temp[i] = distribution(generator) * randomSD; }
	Cuda::CUDA_Array_CopyHostToDevice(temp, sizeWeight * sizeof(double), _weight[0]);
	delete temp;

	_parameterWeight.Set(sizeWeight, MaxBatchSize() * _convHeight * _convWidth);

	if (bias)
	{
		int sizeBias = _convFilter;
		_bias[0] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[1] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[2] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[3] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[4] = Cuda::CUDA_Array_Allocate<double>(sizeBias);

		Cuda::CUDA_Array_Initialize(_bias[0], 0.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[1], 1.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[2], 1.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[3], 0.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[4], 0.0, sizeBias);

		_parameterBias.Set(sizeBias, MaxBatchSize() * _convHeight * _convWidth);
	}
}

CalculationC_2D::~CalculationC_2D()
{
	Cuda::CUDA_Array_Free((void**)&_output);
	Cuda::CUDA_Array_Free((void**)&_signal);

	for (int i = 0; i < 5; i++)
	{
		Cuda::CUDA_Array_Free((void**)&_weight[i]);
		Cuda::CUDA_Array_Free((void**)&_bias[i]);
	}

	delete _source;
}

int CalculationC_2D::Size()
{
	return _convFilter * _convHeight * _convWidth;
}

int CalculationC_2D::ConvRowCount()
{
	return _convHeight;
}

int CalculationC_2D::ConvRowSize()
{
	return _convWidth;
}

int CalculationC_2D::ConvFilter()
{
	return _convFilter;
}

void CalculationC_2D::DropoutSelect() {}

void CalculationC_2D::Forward(int sets, bool training)
{
	_source->Forward(sets, training);

	Calculation::Calculation_Convolution_2D_Forward(
		sets, _source->ConvFilter(), _source->ConvRowCount(), _source->ConvRowSize(),
		_source->Output(), _windowHeight, _windowWidth, _weight[0], _bias[0], _activationType,
		_convFilter, _convHeight, _convWidth, _output);
}

void CalculationC_2D::Backward(int sets, int* dropoutSelect)
{
	Calculation::Calculation_Convolution_2D_Backward(
		sets, _source->ConvFilter(), _source->ConvRowCount(), _source->ConvRowSize(),
		_source->Signal(), _source->Output(), _windowHeight, _windowWidth, _weight, _bias,
		_activationType, _convFilter, _convHeight, _convWidth, _output, _signal,
		_parameterBias, _parameterWeight);

	_source->Backward(sets, dropoutSelect);
}


Pooling_1D::Pooling_1D(Layer * source, int pooling, Global::PoolingType poolingType)
{
	_source = source;
	_sizePooling = pooling;
	_poolingType = poolingType;
	_convRowSize = (_source->ConvRowSize() + _sizePooling - 1) / _sizePooling;

	_pooling = Cuda::CUDA_Array_Allocate<double>(_source->ConvRowSize() * _source->ConvRowCount() * _source->ConvFilter() * MaxBatchSize());
	_output = Cuda::CUDA_Array_Allocate<double>(_convRowSize * _source->ConvRowCount() * _source->ConvFilter() * MaxBatchSize());
	_signal = Cuda::CUDA_Array_Allocate<double>(_convRowSize * _source->ConvRowCount() * _source->ConvFilter() * MaxBatchSize());
}

Pooling_1D::~Pooling_1D()
{
	Cuda::CUDA_Array_Free((void**)&_output);
	Cuda::CUDA_Array_Free((void**)&_signal);
	Cuda::CUDA_Array_Free((void**)&_pooling);

	delete _source;
}

int Pooling_1D::Size()
{
	return _source->ConvFilter() * _source->ConvRowCount() * _convRowSize;
}

int Pooling_1D::ConvRowCount()
{
	return _source->ConvRowCount();
}

int Pooling_1D::ConvRowSize()
{
	return _convRowSize;
}

int Pooling_1D::ConvFilter()
{
	return _source->ConvFilter();
}

void Pooling_1D::DropoutSelect() {}

void Pooling_1D::Forward(int sets, bool training)
{
	_source->Forward(sets, training);

	Calculation::Calculation_Pooling_1D_Forward(
		sets, _source->ConvRowCount(), _source->ConvFilter(), _source->ConvRowSize(),
		_source->Output(), _sizePooling, _convRowSize, _poolingType, _output, _pooling);
}

void Pooling_1D::Backward(int sets, int * dropoutSelect)
{
	Calculation::Calculation_Pooling_1D_Backward(
		sets, _source->ConvRowCount(), _source->ConvFilter(), _source->ConvRowSize(),
		_source->Signal(), _sizePooling, _convRowSize, _signal, _pooling);

	_source->Backward(sets, nullptr);
}


Pooling_2D::Pooling_2D(Layer* source, int poolingHeight, int poolingWidth, Global::PoolingType poolingType)
{
	_source = source;
	_poolingHeight = poolingHeight;
	_poolingWidth = poolingWidth;
	_poolingType = poolingType;
	_convHeight = (_source->ConvRowCount() + _poolingHeight - 1) / _poolingHeight;
	_convWidth = (_source->ConvRowSize() + _poolingWidth - 1) / _poolingWidth;

	_pooling = Cuda::CUDA_Array_Allocate<double>(_source->ConvRowSize() * _source->ConvRowCount() * _source->ConvFilter() * MaxBatchSize());
	_output = Cuda::CUDA_Array_Allocate<double>(_convHeight * _convWidth * _source->ConvFilter() * MaxBatchSize());
	_signal = Cuda::CUDA_Array_Allocate<double>(_convHeight * _convWidth * _source->ConvFilter() * MaxBatchSize());
}

Pooling_2D::~Pooling_2D()
{
	Cuda::CUDA_Array_Free((void**)&_pooling);
	Cuda::CUDA_Array_Free((void**)&_output);
	Cuda::CUDA_Array_Free((void**)&_signal);

	delete _source;
}

int Pooling_2D::Size()
{
	return _source->ConvFilter() * _convHeight * _convWidth;
}

int Pooling_2D::ConvRowCount()
{
	return _convHeight;
}

int Pooling_2D::ConvRowSize()
{
	return _convWidth;
}

int Pooling_2D::ConvFilter()
{
	return _source->ConvFilter();
}

void Pooling_2D::DropoutSelect() {}

void Pooling_2D::Forward(int sets, bool training)
{
	_source->Forward(sets, training);

	Calculation::Calculation_Pooling_2D_Forward(
		sets, _source->ConvFilter(), _source->ConvRowCount(), _source->ConvRowSize(),
		_source->Output(), _poolingHeight, _poolingWidth, _convHeight, _convWidth, _poolingType, _output, _pooling);
}

void Pooling_2D::Backward(int sets, int * dropoutSelect)
{
	Calculation::Calculation_Pooling_2D_Backward(
		sets, _source->ConvFilter(), _source->ConvRowCount(), _source->ConvRowSize(),
		_source->Signal(), _poolingHeight, _poolingWidth, _convHeight, _convWidth, _signal, _pooling);

	_source->Backward(sets, nullptr);
}


CalculationFC::CalculationFC(Layer * source, bool bias, double dropoutActive, Global::ActivationFunction activationType, int oSize)
{
	_source = source;
	_size = oSize;
	_activationType = activationType;

	_dropoutScale = 1.0;
	if (dropoutActive >= 0.0 && dropoutActive <= 1.0)
	{
		_dropoutSelectCount = (int)((double)_size * dropoutActive);
		_dropoutScale = (double)_dropoutSelectCount / (double)Size();
		_dropoutSelectIndex = Cuda::CUDA_Array_Allocate<int>(Size());

		_dropoutTrack = new int[Size()];
		_dropoutCharges = 0;

		if (_dropoutSelectIndex != nullptr)
		{
			_dropoutCharges = 5 * Size();
			for (int i = 0; i < Size(); i++)
			{
				_dropoutTrack[i] = 5;
			}
		}
	}
	else
	{
		_dropoutSelectIndex = nullptr;
	}

	_output = Cuda::CUDA_Array_Allocate<double>(_size * MaxBatchSize());
	_signal = Cuda::CUDA_Array_Allocate<double>(_size * MaxBatchSize());

	int sizeWeight = _size * _source->Size();
	_weight[0] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[1] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[2] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[3] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);
	_weight[4] = Cuda::CUDA_Array_Allocate<double>(sizeWeight);

	Cuda::CUDA_Array_Initialize(_weight[1], 1.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[2], 1.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[3], 0.0, sizeWeight);
	Cuda::CUDA_Array_Initialize(_weight[4], 0.0, sizeWeight);

	default_random_engine generator(random_device{}());
	mt19937 e2(generator());
	normal_distribution<double> distribution(0.0, 1.0);
	double* temp = new double[sizeWeight];
	double randomSD = 1.0 / sqrt(_source->Size());
	for (int i = 0; i < sizeWeight; i++) { temp[i] = distribution(generator) * randomSD; }
	Cuda::CUDA_Array_CopyHostToDevice(temp, sizeWeight * sizeof(double), _weight[0]);
	delete temp;

	_parameterWeight.Set(sizeWeight, MaxBatchSize());
	_parameterSignal.Set(MaxBatchSize() * _source->Size(), _size);

	if (bias)
	{
		int sizeBias = _size;
		_bias[0] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[1] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[2] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[3] = Cuda::CUDA_Array_Allocate<double>(sizeBias);
		_bias[4] = Cuda::CUDA_Array_Allocate<double>(sizeBias);

		Cuda::CUDA_Array_Initialize(_bias[0], 0.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[1], 1.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[2], 1.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[3], 0.0, sizeBias);
		Cuda::CUDA_Array_Initialize(_bias[4], 0.0, sizeBias);

		_parameterBias.Set(sizeBias, MaxBatchSize());
	}

	_parameterForward.Set(MaxBatchSize() * _size, _source->Size());
}

CalculationFC::~CalculationFC()
{
	Cuda::CUDA_Array_Free((void**)&_output);
	Cuda::CUDA_Array_Free((void**)&_signal);

	for (int i = 0; i < 5; i++)
	{
		Cuda::CUDA_Array_Free((void**)&_weight[i]);
		Cuda::CUDA_Array_Free((void**)&_bias[i]);
	}

	Cuda::CUDA_Array_Free((void**)&_dropoutSelectIndex);

	if (_dropoutTrack) { delete _dropoutTrack; }
	delete _source;
}

int CalculationFC::Size()
{
	return _size;
}

int CalculationFC::ConvRowCount()
{
	throw "Asking for convolution info from fully connected layer";
	return 0;
}

int CalculationFC::ConvRowSize()
{
	throw "Asking for convolution info from fully connected layer";
	return 0;
}

int CalculationFC::ConvFilter()
{
	throw "Asking for convolution info from fully connected layer";
	return 0;
}

void CalculationFC::DropoutSelect()
{
	if (_dropoutSelectIndex)
	{
		std::default_random_engine generator(std::random_device{}());
		std::mt19937 e2(generator());
		std::uniform_int_distribution<> distribution(0, _size - 1);
		std::uniform_int_distribution<> check(0, INT_MAX);

		int* tempHostArrayInt = new int[_size];

		for (int i = 0; i < _size; i++) { tempHostArrayInt[i] = 0; }

		for (int i = 0; i < _dropoutSelectCount; i++)
		{
			int temp;
			do
			{
				temp = distribution(e2);
			} while (tempHostArrayInt[temp] == 1 || check(e2) % _dropoutTrack[temp] < 2);
			tempHostArrayInt[temp] = 1;
			_dropoutTrack[temp]--;
		}
		_dropoutCharges -= _dropoutSelectCount;

		while (_dropoutCharges < _size * 5)
		{
			for (int i = 0; i < _size; i++)
			{
				_dropoutTrack[i]++;
			}
			_dropoutCharges += _size;
		}

		Cuda::CUDA_Array_CopyHostToDevice(tempHostArrayInt, _size * sizeof(int), _dropoutSelectIndex);
		delete tempHostArrayInt;
	}

	_source->DropoutSelect();
}

void CalculationFC::Forward(int sets, bool training)
{
	_source->Forward(sets, training);

	Calculation::Calculation_FullyConnected_Forward(
		sets, _source->Size(), _source->Output(),
		training ? _dropoutSelectIndex : nullptr, 
		training ? 1.0 : _dropoutScale, 
		_weight, _bias, _activationType, _size, _output, _parameterForward);
}

void CalculationFC::Backward(int sets, int * dropoutSelect)
{
	Calculation::Calculation_FullyConnected_Backward(
		sets, _source->Size(), _source->Signal(), _source->Output(),
		_weight, _bias, _activationType, _size, _output, dropoutSelect,
		_signal, _parameterBias, _parameterSignal, _parameterWeight);

	_source->Backward(sets, _dropoutSelectIndex);
}
