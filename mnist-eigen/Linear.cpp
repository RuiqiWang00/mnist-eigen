#include "Linear.h"

void Linear::forward(Mat& input, bool required_grad)
{
	if (required_grad)
	{
		for (int i = 0; i < output_len;i++)
		{
			grad.block(i, input_len * i, 1, input_len) = input.row(0);
		}
	}
	input = (input * weight);
	for (int i = 0; i < input.rows(); i++) input.row(i) += bias.row(0);
}

void Linear::backward(Mat& input)
{
	true_grad += input * grad;
	bias_true_grad += input;
	input *= weight.transpose();
	grad.setZero();
}


void Linear::step(double lr)
{
	weight.resize(1, input_len * output_len);
	weight -= true_grad * lr;
	weight.resize(input_len, output_len);
	bias -= bias_true_grad * lr;
	true_grad.setZero();
	bias_true_grad.setZero();
}