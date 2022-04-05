#include "Dataloader.h"

Dataloader::Dataloader(string& data, string& label, int num_data, int len):\
	data(data,ios::in|ios::binary),\
	label(label,ios::in|ios::binary),\
	num_data(num_data),len(len)
{

}

Dataloader::~Dataloader()
{
	data.close();
	label.close();
}

Mat Dataloader::get_data(int i,int length)
{
	long size = length * len;
	data.seekg(16 +  size* i, ios::beg);
	Matrix<unsigned char, Dynamic, Dynamic, RowMajor> m(length,len);
	data.read((char*)m.data(), size);
	return Mat(m.cast<double>());
}

vector<unsigned char> Dataloader::get_label(int i,int length)
{
	long size = length;
	label.seekg(8 + i * size, ios::beg);
	vector<unsigned char> labels(length);
	unsigned char* p = &labels[0];
	label.read((char*)p, 1);
	
	return labels;
}
