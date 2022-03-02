#include "omp.h"

void heat_equation_crank_nicolson(heat_task task, double * v);
void mainsolver(double *udiag, double *mdiag, double *ddiag, double *f, double*x, int n);
void TridiagonalSolver(int n, double *udiag, double *mdiag,  double *ddiag, double *f, double *x);
void BlockTriagonalSolverD(int first,int last, double *udiag, double *mdiag, double *ddiag, double *f);
void BlockTriagonalSolverU(int first,int last, double *udiag, double *mdiag, double *ddiag, double *f);
void Xfinder(int first,int last, double *udiag, double *mdiag, double *ddiag, double *f, double *x,int n);

void heat_equation_crank_nicolson(heat_task task, double * v)
{
	double t,h;
	t=task.T/task.m;
	h=task.L/task.n;
	double *V,*X,*T,*F;
	V=new double[(task.n+1)*(task.m+1)];
	F=new double[(task.n+1)*(task.m+1)];
	X=new double[task.n+1];
	T=new double[task.m+1];

	for (int i=0;i<task.n+1;++i)
		X[i]=i*h;
	for (int i=0;i<task.m+1;++i)
		T[i]=i*t;
	for (int i=0;i<task.m+1;++i)	
	{
		V[i*(task.n+1)]=task.left_condition(T[i]);
		V[i*(task.n+1)+(task.n)]=task.right_condition(T[i]);
	}
	for (int i=0;i<task.n+1;++i)
		V[i]=task.initial_condition(X[i]);
		
	for (int i=0;i<task.m+1;++i)
		for(int j=0;j<task.n+1;++j)
			F[i*(task.n+1)+j]=task.f(j*h,(i+0.5)*t);

	for (int i=1;i<task.m+1;++i)
	{

	double *u,*m,*d,*x,*fi;
	int N=task.n-1;
	u=new double[N];
	m=new double[N];
	d=new double[N];
	x=new double[N];
	fi=new double[N];
	for (int j=0;j<N;++j)
		{
			m[j]=(1+t/(h*h));
			u[j]=d[j]=-t/(2*h*h);
			fi[j]=(1-t/(h*h))*V[(i-1)*(task.n+1)+j+1] + (t/(2*h*h))*(V[(i-1)*(task.n+1)+j] + V[(i-1)*(task.n+1)+j+2]) + t*F[(i-1)*(task.n+1)+j+1];
		}
	fi[0]+=(t/(2*h*h))*V[i*(task.n+1)];
	fi[N-1]+=(t/(2*h*h))* V[i*(task.n+1)+task.n];

	TridiagonalSolver(N, u, m, d, fi, x);
	for (int j=1;j<N+1;++j)
		{
			V[i*(task.n+1)+j]=x[j-1];
		}

	delete[] u;
	delete[] m;
	delete[] d;
	delete[] x;
	delete[] fi;
	}

	for (int i=task.n;i>-1;--i)
		v[i]=V[task.m*(task.n+1)+i];

	delete[] V;
	delete[] F;
	delete[] T;
	delete[] X;
}

void TridiagonalSolver (int n, double *udiag, double *mdiag, double *ddiag, double *f, double *x)
{
	double tmp;
	for (int i = 1; i < n; ++i)
	{
		tmp = ddiag[i]/mdiag[i-1];
		mdiag[i] = mdiag[i] - tmp*udiag[i-1];
		f[i] = f[i] - tmp*f[i-1];
	}

	x[n-1] = f[n-1]/mdiag[n-1];

	for (int i = n - 2; i >= 0; i--)
    {
		x[i]=(f[i]-udiag[i]*x[i+1])/mdiag[i];
	}
}

void BlockTriagonalSolverD(int first,int last, double *udiag, double *mdiag, double *ddiag, double *f)
{
	double tmp;
	for (int i = first+1; i < last; ++i)
	{
		tmp = ddiag[i]/mdiag[i-1];
		mdiag[i] = mdiag[i] - tmp*udiag[i-1];
		f[i] = f[i] - tmp*f[i-1];
		if (first==0)
			ddiag[i]=0;
		else
			ddiag[i]=-tmp*ddiag[i-1];
	}
}

void BlockTriagonalSolverU(int first,int last, double *udiag, double *mdiag, double *ddiag, double *f, int n)
{
	double tmp;
	for (int i=last-1; i>first;--i)
	{
		tmp = udiag[i-1]/mdiag[i];
		ddiag[i-1]-=tmp*ddiag[i];
		f[i-1]=f[i-1] - tmp*f[i];
		if (last==n)
			udiag[i-1]=0;
		else
			udiag[i-1]=-tmp*udiag[i];
	}
}

void Xfinder(int first,int last, double *udiag, double *mdiag, double *ddiag, double *f, double *x,int n)
{
	if (first==0)
		for (int i=first+1; i<last-1;++i)
		{
			x[i]=(f[i]-udiag[i]*x[last+1])/mdiag[i];
		}
	else if (last==n)
		for (int i=first+1; i<last-1;++i)
		{
			x[i]=(f[i]-ddiag[i]*x[first-1])/mdiag[i];
		}
	else
		for (int i=first+1; i<last-1;++i)
		{
			x[i]=(f[i]-ddiag[i]*x[first-1]-udiag[i]*x[last+1])/mdiag[i];
		}
}