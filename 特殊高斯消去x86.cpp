#include<iostream>
#include<fstream>
#include<string>
#include<cstring>
#include<sstream>
#include<map>
#include<windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
using namespace std;

const int maxsize = 3000;
const int maxrow = 60000; //3000*32>90000 ,最多存贮列数90000的被消元行矩阵60000行
const int numBasis = 100000;   //最多存储90000*100000的消元子

//map<int, int*>iToBasis;    //首项为i的消元子的映射
map<int, int*>ans;			//答案

fstream RowFile("被消元行7.txt", ios::in | ios::out);
fstream BasisFile("消元子7.txt", ios::in | ios::out);


int gRows[maxrow][maxsize];   //被消元行最多60000行，3000列
int gBasis[numBasis][maxsize];  //消元子最多40000行，3000列

int ifBasis[numBasis] = { 0 };

void reset() {
	//	read = 0;
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	memset(ifBasis, 0, sizeof(ifBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("被消元行7.txt", ios::in | ios::out);
	BasisFile.open("消元子7.txt", ios::in | ios::out);
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
	RowFile.open("被消元行7.txt", ios::in | ios::out);
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


void SSE_GE()
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
				for (; j + 4 <= maxsize; j += 4) {
					__m128i vij = _mm_loadu_si128((__m128i*) &gRows[i][j]);
					__m128i vj = _mm_loadu_si128((__m128i*) &gBasis[first][j]);
					__m128i vx = _mm_xor_si128(vij, vj);
					_mm_store_si128((__m128i*) &gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 4 <= maxsize; j += 4) {
					__m128i vij = _mm_loadu_si128((__m128i*) &gRows[i][j]);
					_mm_store_si128((__m128i*) &gBasis[first][j], vij);
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

void SSE_8_GE()
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
				for (; j + 8 <= maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) &gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) &gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_store_si256((__m256i*) &gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 8 <= maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) &gRows[i][j]);
					_mm256_store_si256((__m256i*) &gBasis[first][j], vij);
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
	ofstream out("消元结果7.txt");
	ofstream out1("消元结果(4)7.txt");
	ofstream out2("消元结果(8)7.txt");
	long long head,tail,freq;
	reset();
	readBasis();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    GE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << "传统串行算法:" << (tail - head) * 1000.0 / freq << "ms"<< endl ;
	writeResult(out);

	reset();
	readBasis();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    SSE_GE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << "4路并行化算法:" << (tail - head) * 1000.0 / freq << "ms"<< endl ;
	writeResult(out1);

    reset();
	readBasis();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    SSE_8_GE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << "8路并行化算法:" << (tail - head) * 1000.0 / freq << "ms"<< endl ;
	writeResult(out2);

	out.close();
	out1.close();
	out2.close();
	return 0;
}

