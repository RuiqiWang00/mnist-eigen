#include "CrossEntropy.h"

double CrossEntropy::forward(Mat& input, vector<unsigned char>& label, bool required_grad)
{
	if (required_grad)
	{
		grad.setZero();
		grad(0, label[0]) = -1.0 / (input(0, label[0]) + 0.000000001);
	}
	double loss = 0.0;
	int size = input.rows();
	for (int i = 0; i < size; i++)
	{
		loss -= std::log(input(i, label[i]) + 0.000000001);
	}
	return loss / static_cast<double>(size);
}

