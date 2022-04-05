#pragma once
#include "Module.h"
class Loss :
    public Module
{
public:
    Loss(int input_len, int output_len) :Module(input_len, output_len) {};
    virtual ~Loss() {};
    virtual double forward(Mat& input, vector<unsigned char>& label, bool required_grad=true) { return -1; };
    virtual Mat backward(int batch_size) { return grad / batch_size; };
};

