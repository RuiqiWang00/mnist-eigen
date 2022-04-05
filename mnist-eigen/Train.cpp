#include "Train.h"

Train::Train(int batch_size, double lr, int epoches, LSP l, MSP m, DSP train_d,DSP test_d):\
	batch_size(batch_size),\
	lr(lr),\
	epoches(epoches),\
	loss(l),\
	module(m),\
	train_loader(train_d),\
	test_loader(test_d)
{

}


int Train::get_correct(Mat& m, vector<unsigned char>& label) const
{
	int correct = 0;
	int size = label.size();
	for (int k = 0; k <size ; k++)
	{
		MatrixXd::Index maxRow, maxCol;
		m.row(k).maxCoeff(&maxRow, &maxCol);
		if (maxCol == label[k]) correct++;
	}
	return correct;
}


void Train::run()
{
	int batch_num = train_loader->get_num_data() / batch_size;
	for (int i = 0; i < epoches; i++)
	{
		cout << "epoch:  " << i << endl;
		for (int j = 0; j < batch_num; j++)
		{
			double l = 0.0;
			float correct = 0;
			for (int k = 0; k < batch_size; k++)
			{
				Mat m = train_loader->get_data(j*batch_size+k);
				module->forward(m);
				vector<unsigned char> label = train_loader->get_label(j * batch_size + k);
				l+=loss->forward(m, label);
				//if(k%30==0) cout << "k:  "<<k<<"  loss  " << l/(k+1) << endl;
				auto t = loss->backward(batch_size);
				module->backward(t);
				correct += get_correct(m, label);
			}
			cout << "batch: " << j << endl;
			cout << "loss:   " << l/batch_size << endl;
			cout << "correct:  " << correct/static_cast<float>(batch_size) << "\n\n";
			loss->step(lr);
			module->step(lr);
			if ((j + 1) % 3 == 0) test();
			if ((j + 1) % 400 == 0) lr *= 0.9;
		}
		test();
	}
}

void Train::test()
{
	int batch_num = test_loader->get_num_data()/batch_size;
	double l = 0.0;
	float correct = 0.0;
	module->zero_grad();
	for (int i = 0; i < batch_num; i++)
	{
		for (int j = 0; j < batch_size; j++)
		{
			Mat m = test_loader->get_data(i*batch_size+j);
			module->forward(m,false);
			auto label = test_loader->get_label(i*batch_size+j);
			l += loss->forward(m, label,false);
			correct += get_correct(m, label);
		}

	}
	cout << "test loss:  " << l / (static_cast<float>(batch_num * batch_size)) << endl;
	cout << "test correct:  " << correct / static_cast<float>(batch_num*batch_size) << "\n\n";
}