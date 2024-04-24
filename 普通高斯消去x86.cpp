#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cstdlib>
#include <nmmintrin.h>
#include <windows.h>
using namespace std;
const int N=1000;
float A[1000][1000];
void m_reset()
{
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<i;j++)
            A[i][j]=0;
        A[i][i]=1.0;
        for(int j=i+1;j<N;j++)
            A[i][j]=rand();
    }
    for(int k=0;k<N;k++)
        for(int i=k+1;i<N;i++)
            for(int j=0;j<N;j++)
                A[i][j]+=A[k][j];
}

void serial(int n)//普通串行算法
{
    for(int k=0;k<n;k++)
    {
        for(int j = k+1 ; j < n ; j++)
        {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i = k+1 ; i < n ; i++)
        {
            for(int j = k+1 ; j < n ; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void neon_1(int n)//乘法部分向量化，不对齐
{
    for(int k=0;k<n;k++)
    {
        for(int j = k+1 ; j < n ; j++)
        {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i=k+1;i<n;i++)
        {
            __m128 factor4 = _mm_set_ps1(A[i][k]);
            int j;
            for(j=k+1;j+4<=n;j+=4)
            {
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                vakj =  _mm_mul_ps(vakj,factor4);
                vaij = _mm_sub_ps(vaij,vakj);
                _mm_store_ps(&A[i][j],vaij);
            }
            while(j<n)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
                j++;
            }
            A[i][k] = 0;
        }
    }
}

void neon_2(int n)//除法部分向量化，不对齐
{
    for(int k = 0 ; k < n ; k++)
    {
        int j;
        __m128 vt = _mm_set1_ps(A[k][k]);
        for(j = k+1 ; j+4 <= n ; j+=4)
        {
            __m128 va = _mm_loadu_ps(&A[k][j]);
            va = _mm_div_ps(va,vt);
            _mm_store_ps(&A[k][j],va);
        }
        while(j<n)
        {
            A[k][j] = A[k][j]/A[k][k];
            j++;
        }
        A[k][k] = 1.0;
        for(int i = k+1 ; i < n ; i++)
        {
            for(int j = k+1 ; j < n ; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;

        }
    }
}

void neon_3(int n)//乘法、除法都向量化，不对齐
{
    for(int k=0;k<n;k++)
    {
        int j;
       __m128 vt = _mm_set1_ps(A[k][k]);
        for(j = k+1 ; j+4 <= n ; j+=4)
        {
            __m128 va = _mm_loadu_ps(&A[k][j]);
            va = _mm_div_ps(va,vt);
            _mm_store_ps(&A[k][j],va);
        }
        while(j<n)
        {
            A[k][j] = A[k][j]/A[k][k];
            j++;
        }
        A[k][k] = 1.0;
        for(int i=k+1;i<n;i++)
        {
            __m128 factor4 = _mm_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+4<=n;j+=4)
            {
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                vakj =  _mm_mul_ps(vakj,factor4);
                vaij = _mm_sub_ps(vaij,vakj);
                _mm_store_ps(&A[i][j],vaij);
            }
            while(j<n)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
                j++;
            }
            A[i][k] = 0;
        }
    }
}

void neon_4(int n)//乘法部分向量化，对齐
{
    for(int k=0;k<n;k++)
    {
        for(int j = k+1 ; j < n ; j++)
        {
            A[k][j] = A[k][j]/A[k][k];
        }
        A[k][k] = 1.0;
        for(int i=k+1;i<n;i++)
        {
            __m128 factor4 = _mm_set1_ps(A[i][k]);
            int j;
            int start = k+4-k%4;
            for(j=k+1;j<start && j<n;j++)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
            }
            for(j=start;j+4<=n;j+=4)
            {
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                vakj =  _mm_mul_ps(vakj,factor4);
                vaij = _mm_sub_ps(vaij,vakj);
                _mm_store_ps(&A[i][j],vaij);
            }
            while(j<n)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
                j++;
            }
            A[i][k] = 0;
        }
    }
}

void neon_5(int n)//除法部分向量化，对齐
{
    for(int k = 0 ; k < n ; k++)
    {
        int j;
        int start = k+4-k%4;
        __m128 vt = _mm_set1_ps(A[k][k]);
        for(j=k+1;j<start && j<n;j++)
        {
            A[k][j] = A[k][j]/A[k][k];
        }
        for(j = start ; j+4 <= n ; j+=4)
        {
            __m128 va = _mm_loadu_ps(&A[k][j]);
            va = _mm_div_ps(va,vt);
            _mm_store_ps(&A[k][j],va);
        }
        while(j<n)
        {
            A[k][j] = A[k][j]/A[k][k];
            j++;
        }
        A[k][k] = 1.0;
        for(int i = k+1 ; i < n ; i++)
        {
            for(int j = k+1 ; j < n ; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void neon_6(int n)//乘法、除法都向量化，对齐
{
    for(int k=0;k<n;k++)
    {
        int j;
        int start = k+4-k%4;
        __m128 vt = _mm_set1_ps(A[k][k]);
        for(j=k+1;j<start && j<n;j++)
        {
            A[k][j] = A[k][j]/A[k][k];
        }
        for(j = start ; j+4 <= n ; j+=4)
        {
            __m128 va = _mm_loadu_ps(&A[k][j]);
            va = _mm_div_ps(va,vt);
            _mm_store_ps(&A[k][j],va);
        }
        while(j<n)
        {
            A[k][j] = A[k][j]/A[k][k];
            j++;
        }
        A[k][k] = 1.0;
        for(int i=k+1;i<n;i++)
        {
            __m128 factor4 = _mm_set1_ps(A[i][k]);
            int j;
            int start = k+4-k%4;
            for(j=k+1;j<start && j<n;j++)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
            }
            for(j=start;j+4<=n;j+=4)
            {
                __m128 vaij = _mm_loadu_ps(&A[i][j]);
                __m128 vakj = _mm_loadu_ps(&A[k][j]);
                vakj =  _mm_mul_ps(vakj,factor4);
                vaij = _mm_sub_ps(vaij,vakj);
                _mm_store_ps(&A[i][j],vaij);
            }
            while(j<n)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
                j++;
            }
            A[i][k] = 0;
        }
    }
}

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    int n = 100;
    long long head,tail,freq;
    m_reset();
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    serial(n);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << ":" << (tail - head) * 1000.0 / freq << "ms"<< endl ;

    m_reset();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    neon_1(n);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << ":" << (tail - head) * 1000.0 / freq << "ms"<< endl ;

    m_reset();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    neon_2(n);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << ":" << (tail - head) * 1000.0 / freq << "ms"<< endl ;

    m_reset();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    neon_3(n);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << ":" << (tail - head) * 1000.0 / freq << "ms"<< endl ;

    m_reset();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    neon_4(n);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << ":" << (tail - head) * 1000.0 / freq << "ms"<< endl ;

    m_reset();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    neon_5(n);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << ":" << (tail - head) * 1000.0 / freq << "ms"<< endl ;

    m_reset();

    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    neon_6(n);
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout << ":" << (tail - head) * 1000.0 / freq << "ms"<< endl ;
    return 0;
}
