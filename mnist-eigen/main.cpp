#include "Train.h"
#include "MLP.h"


int main()
{
	string train_data("train-images.idx3-ubyte");
	string train_label("train-labels.idx1-ubyte");
	string test_data("t10k-images.idx3-ubyte");
	string test_label("t10k-labels.idx1-ubyte");
	int num_data = 60000;
	int rows = 28;
	int cols = 28;
	int input_len = 784;
	int output_len = 10;
	int epoches = 1;
	int batch_size = 128;
	double lr = 0.01;

	int test_num_data = 10000;

	vector<int> module_size = { 784,128,64,10 };


	shared_ptr<Dataloader> train_set = make_shared<Dataloader>(train_data, train_label\
		, num_data, input_len);
	shared_ptr<Dataloader> test_set = make_shared<Dataloader>(test_data, test_label\
		, test_num_data, input_len);
	shared_ptr<Module> module = make_shared<MLP>(input_len, output_len,module_size);
	shared_ptr<Loss> loss = make_shared<CrossEntropy>(output_len);

	shared_ptr<Train> train(new Train(batch_size,lr,epoches,loss,module,train_set,test_set));
	train->run();
}


