#include <vector>
#include <iostream>
#include <cmath>
#include "time.h"
#include "omp.h"

void SLE_Solver_CRS_BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x, int& count);

class BICG 
{
public:
	int count;
	BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x) : A(A), b(b), n(A.n)
	{
		count=0;
		tmp = new double[n];
        r = new double[n];
    	p = new double[n];
        z = new double[n];
    	s = new double[n];
		transpose();

	for (int i=0;i<AT.n;++i)
	{
		std::cout<<AT.val[i]<<" - "<<AT.colIndex[i]<<"\n";
	}
		for (int i=0;i<A.n;++i)
	{
		std::cout<<A.val[i]<<" - "<<A.colIndex[i]<<"\n";
	}

		for (int i = 0; i < n; ++i)
		{
            tmp[i] = 0.;
            r[i] = p[i] = z[i] = s[i] = b[i];
        }
		for (int i = 0; i < n; ++i)
		{
            x[i] = 0.;
        }

       	for (count = 0; count < max_iter; ++count)
		{
		    multi(A, z, tmp);
		    double prProd = scalar(p, r);
		    double alpha = prProd / scalar(s, tmp);

		    sum(x, 1., z, alpha);
		    sum(r, 1., tmp, -alpha);

		    if (sqrt(scalar(r,r)) < eps)
			{
			    break;
            }

		    multi(AT, s, tmp);
		    sum(p, 1., tmp, -alpha);

		    double beta = scalar(p, r) / prProd;

		    if (fabs(beta) < 1e-14)
			{
			    break;
            }

		    sum(z, beta, r, 1.);
		    sum(s, beta, p, 1.);
	    }
	}

	~BICG()
	{
        delete[] tmp;
        delete[] r;
        delete[] p;
        delete[] z;
        delete[] s;
    }

private:
	const CRSMatrix& A;
    const double* b;
    int n;
    CRSMatrix AT;
    double* tmp;
    double* r;
    double* p;
    double* z;
    double* s;

	void transpose()
	{
		std::vector<std::vector<int>> indxVectors(A.n);
		std::vector<std::vector<double>> valVectors(A.n);

		for (int i = 0; i < A.n; i++)
		{
			indxVectors[i].reserve(A.nz / A.n);
			valVectors[i].reserve(A.nz / A.n);
		}

		for (int i = 0; i < A.n; i++)
		{
			for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; j++)
			{
				int col = A.colIndex[j];
				indxVectors[col].push_back(i);
				valVectors[col].push_back(A.val[j]);
			}
		}
		AT.n = A.n;
		AT.m = A.n;
		AT.nz = 0;

		AT.rowPtr.reserve(A.n + 1);
		size_t k = 0;
		for (int i = 0; i < A.n; i++)
		{
			k += valVectors[i].size();
		}
		AT.val.reserve(k);
		AT.colIndex.reserve(k);

		for (int i = 0; i < A.n; i++)
		{
			AT.rowPtr.push_back(AT.nz);
			for (int j = 0; j < valVectors[i].size(); j++)
			{
				AT.val.push_back(valVectors[i].at(j));
				AT.colIndex.push_back(indxVectors[i].at(j));
				AT.nz++;
			}
		}
		AT.rowPtr.push_back(AT.nz);

	}

	double scalar(const double* a, const double* b) const
	{
	double result = 0;
		for (int i = 0; i < n; ++i)
		{
		    result += a[i] * b[i];
		}
	    return result;
    }

    void sum(double* x, double a, const double* y, double b) const
	{
        #pragma omp parallel for
	    for (int i = 0; i < n; ++i)
		{
	    	x[i] = a * x[i] + b * y[i];
        }
    }

	void multi(const CRSMatrix& A, const double* x, double* y) const
	{
        #pragma omp parallel for
    	for (int i = 0; i < A.n; ++i)
		{
    		y[i] = 0.0;
    		for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j)
			{
    			y[i] += A.val[j] * x[A.colIndex[j]];
            }
    	}
    }
};


void SLE_Solver_CRS_BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x, int& count) 
{
    BICG X(A, b, eps, max_iter, x);
	count=X.count;
	std::cout<<count;
}