#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cstdlib>
#include <arm_neon.h>
const int N=1000;
float A[1000][1000];
void m_reset() 
{
    for(int i=0;i<N;i++) 
    {
        A[i][i]=1.0;
        for(int j=0;j<i;j++)
            A[i][j] = 0;
        for(int j=i+1;j<N;j++)
            A[i][j] = rand();
    }
    for(int k=0;k<N;k++)
        for(int i=k+1;i<N;i++)
            for(int j=0;j<N;j++)
                A[i][j] += A[k][j];
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
            float32x4_t factor4 = vmovq_n_f32(A[i][k]);
            int j;
            for(j=k+1;j+4<=n;j+=4)
            {
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                vakj = vmulq_f32(vakj,factor4);
                vaij = vsubq_f32(vaij,vakj);
                vst1q_f32(&A[i][j],vaij);
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
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        for(j = k+1 ; j+4 <= n ; j+=4)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
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
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        for(j = k+1 ; j+4 <= n ; j+=4)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
        }
        while(j<n)
        {
            A[k][j] = A[k][j]/A[k][k];
            j++;
        }
        A[k][k] = 1.0;
        for(int i=k+1;i<n;i++)
        {
            float32x4_t factor4 = vmovq_n_f32(A[i][k]);
            int j;
            for(j=k+1; j+4 <= n;j+=4)
            {
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                vakj = vmulq_f32(vakj,factor4);
                vaij = vsubq_f32(vaij,vakj);
                vst1q_f32(&A[i][j],vaij);
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
            float32x4_t factor4 = vmovq_n_f32(A[i][k]);
            int j;
            int start = k+4-k%4;
            for(j=k+1;j<start && j<n;j++)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
            }
            for(j=start;j+4<=n;j+=4)
            {
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                vakj = vmulq_f32(vakj,factor4);
                vaij = vsubq_f32(vaij,vakj);
                vst1q_f32(&A[i][j],vaij);
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
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        for(j=k+1;j<start && j<n;j++)
        {
            A[k][j] = A[k][j]/A[k][k];
        }
        for(j = start ; j+4 <= n ; j+=4)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
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
        float32x4_t vt = vmovq_n_f32(A[k][k]);
        for(j=k+1;j<start && j<n;j++)
        {
            A[k][j] = A[k][j]/A[k][k];
        }
        for(j = start ; j+4 <= n ; j+=4)
        {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va,vt);
            vst1q_f32(&A[k][j],va);
        }
        while(j<n)
        {
            A[k][j] = A[k][j]/A[k][k];
            j++;
        }
        A[k][k] = 1.0;
        for(int i=k+1;i<n;i++)
        {
            float32x4_t factor4 = vmovq_n_f32(A[i][k]);
            int j;
            int start = k+4-k%4;
            for(j=k+1;j<start && j<n;j++)
            {
                A[i][j] = A[i][j] - A[k][j]*A[i][k];
            }
            for(j=start;j+4<=n;j+=4)
            {
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                vakj = vmulq_f32(vakj,factor4);
                vaij = vsubq_f32(vaij,vakj);
                vst1q_f32(&A[i][j],vaij);
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
    int n = 10,step=10;
    struct timespec sts,ets;
    while(n<=1000)
    {
        m_reset();
        timespec_get(&sts, TIME_UTC);
        //to measure
        serial(n);
        timespec_get(&ets, TIME_UTC);
        time_t dsec=ets.tv_sec-sts.tv_sec;
        long dnsec=ets.tv_nsec-sts.tv_nsec;
        if (dnsec<0)
        {
            dsec--;
            dnsec+=1000000000ll;
        }
        printf ("传统串行算法用时：%ld.%09lds\n",dsec,dnsec);
        m_reset();
        timespec_get(&sts, TIME_UTC);
        //to measure
        neon_1(n);
        timespec_get(&ets, TIME_UTC);
        dsec=ets.tv_sec-sts.tv_sec;
        dnsec=ets.tv_nsec-sts.tv_nsec;
        if (dnsec<0)
        {
            dsec--;
            dnsec+=1000000000ll;
        }
        printf ("使用Neon将乘法部分并行化，内存不对齐用时:%ld.%09lds\n",dsec,dnsec);
        m_reset();
        timespec_get(&sts, TIME_UTC);
        //to measure
        neon_2(n);
        timespec_get(&ets, TIME_UTC);
        dsec=ets.tv_sec-sts.tv_sec;
        dnsec=ets.tv_nsec-sts.tv_nsec;
        if (dnsec<0)
        {
            dsec--;
            dnsec+=1000000000ll;
        }
        printf ("使用Neon将除法部分并行化，内存不对齐用时:%ld.%09lds\n",dsec,dnsec);
        m_reset();
        timespec_get(&sts, TIME_UTC);
        //to measure
        neon_3(n);
        timespec_get(&ets, TIME_UTC);
        dsec=ets.tv_sec-sts.tv_sec;
        dnsec=ets.tv_nsec-sts.tv_nsec;
        if (dnsec<0)
        {
            dsec--;
            dnsec+=1000000000ll;
        }
        printf ("使用Neon将乘法除法部分都并行化，内存不对齐用时:%ld.%09lds\n",dsec,dnsec);
        m_reset();
        timespec_get(&sts, TIME_UTC);
        //to measure
        neon_4(n);
        timespec_get(&ets, TIME_UTC);
        dsec=ets.tv_sec-sts.tv_sec;
        dnsec=ets.tv_nsec-sts.tv_nsec;
        if (dnsec<0)
        {
            dsec--;
            dnsec+=1000000000ll;
        }
        printf ("使用Neon将乘法部分并行化，内存对齐用时:%ld.%09lds\n",dsec,dnsec);
        m_reset();
        timespec_get(&sts, TIME_UTC);
        //to measure
        neon_5(n);
        timespec_get(&ets, TIME_UTC);
        dsec=ets.tv_sec-sts.tv_sec;
        dnsec=ets.tv_nsec-sts.tv_nsec;
        if (dnsec<0)
        {
            dsec--;
            dnsec+=1000000000ll;
        }
        printf ("使用Neon将除法部分并行化，内存对齐用时:%ld.%09lds\n",dsec,dnsec);
        m_reset();
        timespec_get(&sts, TIME_UTC);
        //to measure
        neon_6(n);
        timespec_get(&ets, TIME_UTC);
        dsec=ets.tv_sec-sts.tv_sec;
        dnsec=ets.tv_nsec-sts.tv_nsec;
        if (dnsec<0)
        {
            dsec--;
            dnsec+=1000000000ll;
        }
        printf ("使用Neon将乘法除法部分都并行化，内存对齐用时:%ld.%09lds\n",dsec,dnsec);
        n+=step;
        if(n == 100)
            step = 100;
    }
    return 0;
}