#include "Softmax.h"

void Softmax::forward(Mat& input, bool required_grad=true)
{
	input = input.array().exp();
	input.array().colwise() /= input.rowwise().sum().array();

	if (required_grad)
	{
		grad=-(input.row(0).transpose()*input.row(0));
		grad.diagonal() = input.row(0).array() * (1.0 - input.row(0).array());
		//cout << "softmax grad :  " << grad << "\n\n";
	}
}