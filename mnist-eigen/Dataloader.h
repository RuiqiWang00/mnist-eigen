#pragma once
#include "util.h"
#include <string>
#include <fstream>
using namespace std;
class Dataloader
{
public:
	Dataloader(string& data, string& label, int num_data, int len);
	~Dataloader();
	Mat get_data(int i,int length=1);
	vector<unsigned char> get_label(int i,int length=1);
	int get_num_data() { return num_data; };
private:
	int num_data;
	int len;
	ifstream data;
	ifstream label;
};

