#pragma once
#include "util.h"
#include <iostream>
class Module
{
public:
	Module(int input_len,int output_len):input_len(input_len),\
		output_len(output_len),\
		grad(Mat::Zero(output_len,input_len)) {};
	virtual ~Module() {};
	virtual void forward(Mat& input, bool required_grad= true) {};
	virtual void backward(Mat& input) {   input = input * grad; grad.setZero(); };
	virtual void step(double lr) { grad.setZero(); };
	virtual void zero_grad() { grad.setZero(); };
protected:
	int input_len;
	int output_len;
	Mat grad;
};
