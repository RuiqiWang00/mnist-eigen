#include "MLP.h"
#include <iostream>

MLP::MLP(int input_len, int output_len, vector<int>& module_size):\
	Module(input_len,output_len)
{
	int size = module_size.size();
	for (int i = 1; i < size; i++)
	{
		list.emplace_back(make_shared<Linear>(module_size[i - 1], module_size[i]));
		if (i == size - 1)
		{
			list.emplace_back(make_shared<Softmax>(module_size[i]));
		}
		else list.emplace_back(make_shared<Relu>(module_size[i]));
	}
}

void MLP::forward(Mat& input, bool required_grad=true)
{
	for (auto m : list)
	{
		m->forward(input, required_grad);
	}
}

void MLP::backward(Mat& input)
{
	for (int i = (list.size() - 1); i >= 0; i--)
	{
		list[i]->backward(input);
	}
}

void MLP::step(double lr)
{
	for (auto m : list)
	{
		m->step(lr);
	}
}

void MLP::zero_grad()
{
	for (auto m : list)
	{
		m->zero_grad();
	}
}
