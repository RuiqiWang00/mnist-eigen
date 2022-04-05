#pragma once
#include "Module.h"
#include "Softmax.h"
#include "Linear.h"
#include "Relu.h"
#include "util.h"
class MLP :
    public Module
{
public:
    MLP(int input_len, int output_len, vector<int>& module_size);
    virtual ~MLP() {};
    virtual void forward(Mat& input, bool required_grad) override;
    virtual void backward(Mat& input);
    virtual void step(double lr);
    virtual void zero_grad();
    using MSP = shared_ptr<Module>;
    using Modulelist = vector<MSP>;
private:
    Modulelist list;
};

