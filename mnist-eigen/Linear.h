#pragma once
#include "Module.h"
#include <iostream>
class Linear :
    public Module
{
public:
    Linear(int input_len,int output_len):Module(input_len,output_len),\
        weight(Mat::Random(input_len, output_len)/static_cast<double>(input_len)),\
        bias(Mat::Random(1,output_len)/ static_cast<double>(output_len)),\
        true_grad(Mat::Zero(1,input_len*output_len)),\
        bias_true_grad(Mat::Zero(1,output_len)) 
    {
        grad = Mat::Zero(output_len, output_len * input_len);
       // cout << "weight" << weight << "\n\n";
        //cout << "bias" << bias << "\n\n";
    };
    virtual ~Linear() {};
    virtual void forward(Mat& input, bool required_grad=true) override;
    virtual void backward(Mat& input) override;
    virtual void step(double lr) override;
private:
    Mat weight;
    Mat bias;
    Mat true_grad;
    Mat bias_true_grad;
};

