#pragma once

namespace Global
{
	enum ActivationFunction
	{
		TanH = 0,
		Sigmoid = 1,
		ReLu = 2,
		LeakyReLu = 3
	};

	enum PoolingType
	{
		AbsoluteMax = 0,
		Average = 1
	};

	enum ErrorFunction
	{
		MeanSquare = 0,
		CrossEntropy = 1
	};

	enum UpdateMethod
	{
		SGD = 0,
		Adam = 1
	};

	enum LayerType
	{
		CalculationC_1D,
		Pooling_1D,
		CalculationC_2D,
		Pooling_2D,
		CalculationFC
	};
}
