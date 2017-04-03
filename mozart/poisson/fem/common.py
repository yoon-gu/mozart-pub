from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

def nJacobiP(x, alpha=0, beta=0, degree=0):
	"""
	Evaluate normalized Jacobi polynomial of type alpha, beta > -1 at point x for order n
	(the special case of alpha = beta = 0, knows as the normalized Legendre polynomial)

	Paramters
		- ``x`` (``float64 array``) : variable x
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``P`` (``float64 array``) : the value of degree-th order normalized Jacobi polynomial at x
	
	Example
		>>> N = 2
		>>> x = np.array([-1, 0, 1])
		>>> from mozart.poisson.fem.common import nJacobiP
		>>> P = nJacobiP(x,0,0,N)
		>>> print(P)
		array([ 1.58113883, -0.79056942,  1.58113883])	
	"""
	Pn = np.zeros((degree+1,x.size),float)
	Pn[0,:] = np.sqrt(2.0**(-alpha-beta-1) * np.math.gamma(alpha+beta+2) / ((np.math.gamma(alpha + 1) * np.math.gamma(beta + 1))))

	if degree == 0:
		P = Pn
	else:
		Pn[1,:] = np.multiply(Pn[0,:]*np.sqrt((alpha+beta+3.0)/((alpha+1)*(beta+1))),((alpha+beta+2)*x+(alpha-beta)))/2
		a_n = 2.0/(2+alpha+beta)*np.sqrt((alpha+1.0)*(beta+1.0)/(alpha+beta+3.0))
		for n in range(2,degree+1):
			anew=2.0/(2*n+alpha+beta)*np.sqrt(n*(n+alpha+beta)*(n+alpha)*(n+beta)/((2*n+alpha+beta-1.0)*(2*n+alpha+beta+1.0)))
			b_n=-(alpha**2-beta**2)/((2*(n-1)+alpha+beta)*(2*(n-1)+alpha+beta+2.0))
			Pn[n,:]=(np.multiply((x-b_n),Pn[n-1,:])-a_n*Pn[n-2,:])/anew
			a_n=anew

	P = Pn[degree,:]
	return P

def DnJacobiP(x, alpha=0, beta=0, degree=0):
	"""
	Evaluate the derivative of the normalized Jacobi polynomial of type alpha, beta > -1 at point x for order n

	Paramters
		- ``x`` (``float64 array``) : variable x
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``dP`` (``float64 array``) : the value of the derivative of the normalized Jacobi polynomial at x according to alpha, beta, degree
	
	Example
		>>> N = 2
		>>> x = np.array([-1, 0, 1])
		>>> from mozart.poisson.fem.common import DnJacobiP
		>>> dP = DnJacobiP(x,0,0,N)
		>>> print(dP)
		array([-4.74341649,  0.        ,  4.74341649])	
	"""
	dP = np.zeros(x.size,float)
	if degree == 0:
		dP[:] = 0
	else:
		dP[:] = np.sqrt(degree*(degree+alpha+beta+1.0))*nJacobiP(x,alpha+1,beta+1,degree-1)
	return dP

def nJacobiGQ(alpha=0, beta=0, degree=0):
	"""
	Compute the degree-th order Gauss quadrature points x and weights w
	associated with the nomalized Jacobi polynomial of type alpha, beta > -1

	Paramters
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : Gauss quadrature points
		- ``w`` (``float64 array``) : Gauss quadrature weights
	
	Example
		>>> N = 2
		>>> from mozart.poisson.fem.common import nJacobiGQ
		>>> x, w = nJacobiGQ(0,0,N)
		>>> print(x)
		array([ -7.74596669e-01,  -4.78946310e-17,   7.74596669e-01])
		>>> print(w)
		array([ 0.55555556,  0.88888889,  0.55555556])
	"""
	if degree == 0:
		x = -(alpha - beta)/(alpha + beta + 2.0)
		w = 2
	else:
		if alpha + beta < 10*np.finfo(float).eps:
			tmp = np.zeros(degree+1)
			tmp[1:] = -(alpha**2-beta**2)/((2.0*np.arange(1,degree+1)+alpha+beta+2.0)*(2.0*np.arange(1,degree+1)+alpha+beta))/2.0
			J = np.diag(tmp) + np.diag(2.0/(2.0*np.arange(1,degree+1)+alpha+beta)*np.sqrt(np.arange(1,degree+1)*(np.arange(1,degree+1)+alpha+beta)* \
					(np.arange(1,degree+1)+alpha)*(np.arange(1,degree+1)+beta)/(2.0*np.arange(1,degree+1)+alpha+beta-1.0)/(2.0*np.arange(1,degree+1)+alpha+beta+1.0)),1)
		else:
			J = np.diag(-(alpha**2-beta**2)/((2.0*np.arange(0,degree+1)+alpha+beta+2.0)*(2.0*np.arange(0,degree+1)+alpha+beta))/2.0)+ \
					np.diag(2.0/(2.0*np.arange(1,degree+1)+alpha+beta)*np.sqrt(np.arange(1,degree+1)*(np.arange(1,degree+1)+alpha+beta)* \
					(np.arange(1,degree+1)+alpha)*(np.arange(1,degree+1)+beta)/(2.0*np.arange(1,degree+1)+alpha+beta-1.0)/(2.0*np.arange(1,degree+1)+alpha+beta+1.0)),1)
		
		J = J + np.transpose(J)

		x, V = np.linalg.eig(J)
		w = np.transpose(V[0])**2 * 2**(alpha + beta + 1) / (alpha + beta + 1.0) * np.math.gamma(alpha + 1.0)  * \
			np.math.gamma(beta + 1.0) / np.math.gamma(alpha + beta + 1.0)
		ind = np.argsort(x)
		x = x[ind]
		w = w[ind]
	return (x, w)

def nJacobiGL(alpha=0, beta=0, degree=0):
	"""
	Compute the degree-th order Gauss Lobatto quadrature points x
	associated with the nomalized Jacobi polynomial of type :math:`\\alpha, \\beta > -1`

	Paramters
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : Gauss Lobatto quadrature points
	
	Example
		>>> N = 3
		>>> from mozart.poisson.fem.common import nJacobiGL
		>>> x = nJacobiGQ(0,0,N)
		>>> print(x)
		array([-1.       , -0.4472136,  0.4472136,  1.       ])
	"""
	if degree == 0:
		x = 0
	elif degree == 1:
		x = np.array([-1, 1])
	else:
		xint, w = nJacobiGQ(alpha+1,beta+1,degree-2)
		x = np.hstack((np.array([-1]),xint))
		x = np.hstack((x,np.array([1])))	
	return x