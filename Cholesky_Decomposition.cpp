#include <math.h>
#include "omp.h"

void Cholesky_Decomposition(double * A, double * L, int n)
{
	for (int i=0;i<n;++i)
		for (int j=0;j<n;++j)
		L[i*n+j]=A[i*n+j];

	int Block_cap=300;
	int coef=n/Block_cap;
	int first,last;
	double sum;
	
	//общие итерации
	for (int i=0;i<coef;++i)
	{
		first=i*Block_cap;
		last=first+Block_cap;
		
	//Обрабатываем A11
	for (int i=first; i<last; ++i)
	{
		for (int k=first; k<i; ++k)
		{
			L[i*n + i] -= L[i*n + k] * L[i*n + k];
		}
		L[i*n + i]=sqrt(L[i*n + i]);

		for (int j=i+1; j<last; ++j)
		{
			for (int k=first; k<i; ++k)
				L[j*n + i] -= L[i*n + k]*L[j*n + k];
			L[j*n + i] /= L[i*n + i];
		}
	}	
	
		//Обработка A21
		for (int i1=first;i1<last;++i1)
		{
		    #pragma omp parallel for
			for(int i2=last;i2<n;++i2)
			{
				sum=0;
				for (int k=first;k<i1;++k)
				{
					sum+=L[i2*n+k]*L[i1*n+k];
				}
			
				L[i2*n+i1] =(L[i2*n+i1] - sum)/L[i1*(n+1)];
			}
		}
		
		//Корректируем A22
		for (int i1=last;i1<n;++i1)
			for(int i2=last;i2<n;++i2)
			{
					sum=0;
					for(int k=first; k<last;++k)
						sum+=L[i1*n+k]*L[i2*n+k];

				L[i1*n+i2]-=sum;
			}
	}
	
	first=coef*Block_cap;
	last=n;
	
	//Обрабатываем остатки
	for (int i=first; i<last; ++i)
	{
		for (int k=first; k<i; ++k)
		{
			L[i*n + i] -= L[i*n + k] * L[i*n + k];
		}
		L[i*n + i]=sqrt(L[i*n + i]);

		for (int j=i+1; j<last; ++j)
		{
			for (int k=first; k<i; ++k)
				L[j*n + i] -= L[i*n + k]*L[j*n + k];
			L[j*n + i] /= L[i*n + i];
		}
	}																	 

	for (int i=0;i<n;++i)
	{
		for(int j=i+1;j<n;++j)
			L[i*n+j]=0;
	}
}