#include<iostream>
#include<fstream>
#include<string>
#include<cstring>
#include<sstream>
#include<map>
#include<arm_neon.h>
#include<stdio.h>
#include<time.h>
using namespace std;

const int maxsize = 3000;
const int maxrow = 60000; //3000*32>90000 ,最多存贮列数90000的被消元行矩阵60000行
const int numBasis = 100000;   //最多存储90000*100000的消元子

//map<int, int*>iToBasis;    //首项为i的消元子的映射
map<int, int*>ans;			//答案

fstream RowFile("被消元行3.txt", ios::in | ios::out);
fstream BasisFile("消元子3.txt", ios::in | ios::out);


int32_t gRows[maxrow][maxsize];   //被消元行最多60000行，3000列
int32_t gBasis[numBasis][maxsize];  //消元子最多40000行，3000列

int ifBasis[numBasis] = { 0 };

void reset() {
	//	read = 0;
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	memset(ifBasis, 0, sizeof(ifBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("被消元行3.txt", ios::in | ios::out);
	BasisFile.open("消元子3.txt", ios::in | ios::out);
	//iToBasis.clear();
	ans.clear();
}

int readBasis() //读取消元子
{          
	for (int i = 0; i < numBasis; i++) {
		if (BasisFile.eof()) {
			cout << "读取消元子" << i - 1 << "行" << endl;
			return i - 1;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(BasisFile, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			//cout << pos << " ";
			if (!flag) {
				row = pos;
				flag = true;
				//iToBasis.insert(pair<int, int*>(row, gBasis[row]));
				ifBasis[row] = 1;
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
	return -1;
}

int readRowsFrom(int pos)  //读取被消元行
{       
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("被消元行3.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   //重置为0
	string line;
	for (int i = 0; i < pos; i++) {       //读取pos前的无关行
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			cout << "读取被消元行 " << i << " 行" << endl;
			return i;   //返回读取的行数
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "read max rows" << endl;
	return -1;  //成功读取maxrow行

}

int findfirst(int row) //寻找第row行被消元行的首项
{  
	int first;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			first = pos + offset;
			return first;
		}
	}
	return -1;
}

void writeResult(ofstream& out) 
{
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first / 32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE() 
{
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	int num = (flag == -1) ? maxrow : flag;
	for (int i = 0; i < num; i++) {
		while (findfirst(i)!= -1) {     //存在首项
			int first =findfirst(i);      //first是首项
			if (ifBasis[first]==1) {  //存在首项为first消元子
				//int* basis = iToBasis.find(first)->second;  //找到该消元子的数组
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //进行异或消元

				}
			}
			else {   //升级为消元子
				//cout << first << endl;
				for (int j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;
			}
		}
	}
}


void AVX_GE() 
{
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	int num = (flag == -1) ? maxrow : flag;
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //存在该消元子
				//int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 4 < maxsize; j += 4) {
					int32x4_t vij = vld1q_s32(&gRows[i][j]);
					int32x4_t vj = vld1q_s32(&gBasis[first][j]);
					int32x4_t vx = veorq_s32(vij, vj);
					vst1q_s32(&gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 4 < maxsize; j += 4) {
					int32x4_t vij = vld1q_s32(&gRows[i][j]);
					vst1q_s32(&gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;

			}
		}
	}
}

int main() 
{
	struct timespec sts,ets;
	ofstream out("消元结果3.txt");
	ofstream out1("消元结果(Neon)3.txt");
	reset();
	readBasis();
	timespec_get(&sts, TIME_UTC);
	// to measure
	GE();
	timespec_get(&ets, TIME_UTC);
	time_t dsec=ets.tv_sec-sts.tv_sec;
	long dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
	dsec--;
	dnsec+=1000000000ll;
	}
	printf ("传统串行算法用时:%ld.%09lds\n",dsec,dnsec);
	writeResult(out);

	reset();
	readBasis();
	timespec_get(&sts, TIME_UTC);
	// to measure
	AVX_GE();
	timespec_get(&ets, TIME_UTC);
	dsec=ets.tv_sec-sts.tv_sec;
	dnsec=ets.tv_nsec-sts.tv_nsec;
	if (dnsec<0){
	dsec--;
	dnsec+=1000000000ll;
	}
	printf ("用AVX并行化用时:%ld.%09lds\n",dsec,dnsec);
	writeResult(out1);
	out.close();
	out1.close();
	return 0;
}

