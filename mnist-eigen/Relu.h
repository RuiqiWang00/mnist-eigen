#pragma once
#include "Module.h"
class Relu :
    public Module
{
public:
    Relu(int input_len) :Module(input_len, input_len) {};
    virtual ~Relu() {};
    virtual void forward(Mat& input, bool required_grad=true);
};


