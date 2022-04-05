#pragma once
#include "Module.h"
class Softmax :
    public Module
{
public:
    Softmax(int input_len) :Module(input_len, input_len) {};
    virtual ~Softmax() {};
    virtual void forward(Mat& input, bool required_grad);
};

