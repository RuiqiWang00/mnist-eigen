#pragma once
#include "Loss.h"
#include <cmath>
class CrossEntropy :
    public Loss
{
public:
    CrossEntropy(int input_len) :Loss(input_len,1) {};
    virtual ~CrossEntropy() {};
    virtual double forward(Mat& input, vector<unsigned char>& label, bool required_grad=true) override;
};

