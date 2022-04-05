#pragma once
#include "Dataloader.h"
#include <iostream>
#include "CrossEntropy.h"
class Train
{
public:
	using LSP = shared_ptr<Loss>;
	using MSP = shared_ptr<Module>;
	using DSP = shared_ptr<Dataloader>;
	Train(int batch_size,double lr,int epoches,LSP,MSP,DSP,DSP);
	~Train(){};
	void run();
	void test();
private:
	int batch_size;
	double lr;
	int epoches;
	LSP loss;
	MSP module;
	DSP train_loader;
	DSP test_loader;
	int get_correct(Mat& m, vector<unsigned char>& label) const;
};

