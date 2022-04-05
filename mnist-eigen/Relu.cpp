#include "Relu.h"

void Relu::forward(Mat& input, bool required_grad)
{
	//input = (input.cwiseAbs() + input) / 2.0;
	input = input.array().max(Mat::Zero(input.rows(), input_len).array());
	if (required_grad)
	{
		grad.diagonal() = input.row(0);
		grad = (grad.array() > 0.0).select(Mat::Identity(output_len, input_len), grad);
	}
}