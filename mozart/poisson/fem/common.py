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

def VandermondeM1D(degree,r):
	"""
	Initialize the 1D Vandermonde matrix, :math:`V_{i,j} = \\phi_j(r_i)`

	Paramters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : points

	Returns
		- ``V1D`` (``float64 array``) : 1D Vandermonde matrix

	Example
		>>> N = 3
		>>> from mozart.poisson.fem.interval import VandermondeM1D
		>>> r = np.linspace(-1,1,N+1)
		>>> V1D = VandermondeM1D(N,r)
		>>> print(V1D)
		array([[ 0.70710678, -1.22474487,  1.58113883, -1.87082869],
		[ 0.70710678, -0.40824829, -0.52704628,  0.76218947],
		[ 0.70710678,  0.40824829, -0.52704628, -0.76218947],
		[ 0.70710678,  1.22474487,  1.58113883,  1.87082869]])
	"""
	V1D = np.zeros((r.size,degree+1),float)
	for j in range(0,degree+1):
		V1D[:,j] = nJacobiP(r,0,0,j)
	return V1D

def DVandermondeM1D(degree, r):
	"""
	Initialize the derivative of modal basis (i) at (r) at order degree

	Paramters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : points

	Returns
		- ``DVr`` (``float64 array``) : Differentiate Vandermonde matrix

	Example
		>>> N = 3
		>>> from mozart.poisson.fem.interval import VandermondeM1D
		>>> r = np.linspace(-1,1,N+1)
		>>> DVr = DVandermondeM1D(N,r)
		>>> print(DVr)
		array([[  0.        ,   1.22474487,  -4.74341649,  11.22497216],
		[  0.        ,   1.22474487,  -1.58113883,  -1.24721913],
		[  0.        ,   1.22474487,   1.58113883,  -1.24721913],
		[  0.        ,   1.22474487,   4.74341649,  11.22497216]])
	"""
	DVr = np.zeros((r.size,degree+1), float)
	for j in range(0,degree+1):
		DVr[:,j] = DnJacobiP(r,0,0,j)
	return DVr

def Dmatrix1D(degree, r, V):
	"""
	Initialize the derivative of modal basis (i) at (r) at order degree

	Paramters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : points

	Returns
		- ``DVr`` (``float64 array``) : Differentiate Vandermonde matrix

	Example
		>>> N = 3
		>>> from mozart.poisson.fem.interval import Dmatrix1D
		>>> r = np.linspace(-1,1,N+1)
		>>> Dr = Dmatirx1D(N,r)
		>>> print(Dr)
		array([[-2.75,  4.5 , -2.25,  0.5 ],
		[-0.5 , -0.75,  1.5 , -0.25],
		[ 0.25, -1.5 ,  0.75,  0.5 ],
		[-0.5 ,  2.25, -4.5 ,  2.75]])
	"""
	Vr = DVandermondeM1D(degree, r)
	Dr = np.linalg.solve(np.transpose(V),np.transpose(Vr))
	Dr = np.transpose(Dr)
	return Dr

def RefNodes_Tri(degree):
	"""
	Computes uniform nodes in the reference triangle for arbitrary polynomial degrees

	Parameters
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle

	Example
		>>> N = 3
		>>> r, s = RefNodes_Tri(N)
		>>> r
		array([-1.        , -0.33333333,  0.33333333,  1.        , -1.        ,
		   -0.33333333,  0.33333333, -1.        , -0.33333333, -1.        ])
		>>> s
		array([-1.        , -1.        , -1.        , -1.        , -0.33333333,
		   -0.33333333, -0.33333333,  0.33333333,  0.33333333,  1.        ])
	"""
	if degree == 0:
		r = np.array([-1.0/3])
		s = np.array([-1.0/3])
	else:
		nrLocal = int((degree + 1)*(degree + 2)/2)
		x = np.linspace(-1, 1, degree + 1)
		r = np.zeros(nrLocal, dtype = np.float64)
		s = np.zeros(nrLocal, dtype = np.float64)
		for j in range (0, degree+1):
			r[int((degree + 1)*j - j*(j-1)/2) + np.arange(0,degree+1-j,1)] = x[np.arange(0,degree+1-j,1)]
			s[int((degree + 1)*j - j*(j-1)/2) + np.arange(0,degree+1-j,1)] = x[j]
	return (r,s)

def rs2ab(r,s):
	"""
	Transfer from (r,s) to (a,b) coordinates in triangle

	Parameters
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle

	Returns
		- ``a`` (``float64 array``) : 2(1+r)/(1-s)-1
		- ``b`` (``float64 array``) : s

	Example
		>>> N = 3
		>>> r, s = RefNodes_Tri(N)
		>>> a, b = rs2ab(r,s)
		>>> a
		array([ -1.00000000e+00,  -3.33333333e-01,   3.33333333e-01,
		   1.00000000e+00,  -1.00000000e+00,  -1.11022302e-16,
		   1.00000000e+00,  -1.00000000e+00,   1.00000000e+00,
		   -1.00000000e+00])
		>>> b
		array([-1.        , -1.        , -1.        , -1.        , -0.33333333,
		   -0.33333333, -0.33333333,  0.33333333,  0.33333333,  1.        ])
	"""
	Np = r.size
	a = np.zeros(Np,float)

	for n in range(0,Np):
		if s[n] != 1:
			a[n] = 2 * (1 + r[n])/(1 - s[n]) - 1
		else:
			a[n] = -1.

	b = s
	return (a,b)

def Simplex2DP(a,b,i,j):
	"""
	Compute 2D orthonormal polynomial on simplex at (a, b) of order (i, j)

	Parameters
		- ``a`` (``folat64 array``) : the value for the first normalized Jacobi polynomial in modal basis
		- ``b`` (``float64 array``) : the value for the second normalized Jacobi polynomial in modal basis
		- ``i`` (``int32``) : order of the the first normalized Jacobi polynomial in modal basis
		- ``j`` (``int32``) : order of the the second normalized Jacobi polynomial in modal basis

	Returns
		- ``P`` (``float64 array``) : evaluated value

	Example
		>>> a = np.array([0, 1])
		>>> b = np.array([2, 3])
		>>> p = Simplex2DP(a, b, 0, 0)
		>>> p
		array([ 0.70710678,  0.70710678])
	"""
	h1 = nJacobiP(a,0,0,i)
	h2 = nJacobiP(b,2*i+1,0,j)
	P = np.sqrt(2.)*h1*h2*(1-b)**i
	return P

def Vandermonde2D(degree,r,s):
	"""
	Initialize the 2D Vandermonde Matrix, :math:`V_{i,j} = \\phi_j(r_i)`

	Parameters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle

	Returns
		- ``V2D`` (``float64 array``) : Vandermonde matrix in 2D

	Example
		>>> N = 2
		>>> r, s = RefNodes_Tri(N)
		>>> V2D = Vandermonde2D(N,r,s)
		>>> V2D
		array([[ 0.70710678, -1.        ,  1.22474487, -1.73205081,  2.12132034,  2.73861279],
		   [ 0.70710678, -1.        ,  1.22474487,  0.        , -0.        , -1.36930639],
		   [ 0.70710678, -1.        ,  1.22474487,  1.73205081, -2.12132034,  2.73861279],
		   [ 0.70710678,  0.5       , -0.61237244, -0.8660254 , -1.59099026,  0.6846532 ],
		   [ 0.70710678,  0.5       , -0.61237244,  0.8660254 ,  1.59099026,  0.6846532 ],
		   [ 0.70710678,  2.        ,  3.67423461, -0.        , -0.        ,  0.        ]])
	"""
	V2D = np.zeros((r.size, int((degree+1) * (degree+2) / 2)), dtype = np.float64)
	a, b = rs2ab(r, s)

	sk = 0
	for i in range(0, degree + 1):
		for j in range(0, degree + 1 - i):
			V2D[:, sk] = Simplex2DP(a,b,i,j)
			sk = sk + 1
	return V2D

def GradSimplex2DP(a,b,id,jd):
	"""
	Return the derivatives of the modal basis (id,jd) on the 2D simplex at (a,b).

	Parameters
		- ``a`` (``float64``) : 2(1+r)/(1-s) - 1
		- ``b`` (``float64``) : s
		- ``id`` (``int32``) : order of the the first normalized Jacobi polynomial in modal basis
		- ``jd`` (``int32``) : order of the the second normalized Jacobi polynomial in modal basis

	Returns
		- ``dmodedr`` (``float64 array``) : derivative value of modal basis on the 2D simplex along r-direction
		- ``dmodeds`` (``float64 array``) : derivative value of modal basis on the 2D simplex along s-direction

	Example
		>>> N = 2
		>>> r, s = RefNodes_Tri(N)
		>>> a, b = rs2ab(r,s)
		>>> dmodedr, dmodeds = GradSimplex2DP(a,b,1,1)
		>>> dmodedr
		array([-2.12132034, -2.12132034, -2.12132034,  3.18198052,  3.18198052,  8.48528137])
		>>> dmodeds
		array([-6.36396103, -1.06066017,  4.24264069, -1.06066017,  4.24264069,  4.24264069])
	"""
	fa = nJacobiP(a, 0, 0, id)
	dfa = DnJacobiP(a, 0, 0, id)
	gb = nJacobiP(b, 2 * id + 1, 0, jd)
	dgb = DnJacobiP(b, 2 * id + 1, 0, jd)

	dmodedr = dfa * gb
	if id > 0:
		dmodedr = dmodedr * ((0.5 * (1 - b))**(id - 1))

	dmodeds = dfa * (gb * (0.5 * (1 + a)))
	if id > 0:
		dmodeds = dmodeds * ((0.5 * (1 - b))**(id - 1))

	tmp = dgb * ((0.5 * (1 - b))**id)

	if id > 0:
		tmp = tmp - 0.5 * id * gb * ((0.5 * (1 - b))**(id - 1))

	dmodeds = dmodeds + fa * tmp
	dmodedr = 2**(id + 0.5) * dmodedr
	dmodeds = 2**(id + 0.5) * dmodeds
	return (dmodedr, dmodeds)

def GradVandermonde2D(degree,r,s):
	"""
	Initialize the gradient of the modal basis (i,j) at (r,s) at order N

	Parameters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle

	Returns
		- ``V2Dr`` (``float64 array``) : Gradient of Vandermonde matrix on the 2D simplex along r-direction
		- ``V2Ds`` (``float64 array``) : Gradient of Vandermonde matrix on the 2D simplex along s-direction

	Example
		>>> N = 2
		>>> r, s = RefNodes_Tri(N)
		>>> V2Dr, V2Ds = GradVandermonde2D(degree,r,s)
		>>> V2Dr
		array([[ 0.        , -0.        ,  0.        ,  1.73205081, -2.12132034, -8.21583836],
		   [ 0.        , -0.        ,  0.        ,  1.73205081, -2.12132034,  0.        ],
		   [ 0.        , -0.        ,  0.        ,  1.73205081, -2.12132034,  8.21583836],
		   [ 0.        ,  0.        , -0.        ,  1.73205081,  3.18198052, -4.10791918],
		   [ 0.        ,  0.        , -0.        ,  1.73205081,  3.18198052,  4.10791918],
		   [ 0.        ,  0.        ,  0.        ,  1.73205081,  8.48528137, -0.        ]])
		>>> V2Ds
		array([[ 0.        ,  1.5       , -4.89897949,  0.8660254 , -6.36396103, -2.73861279],
		   [ 0.        ,  1.5       , -4.89897949,  0.8660254 , -1.06066017,  1.36930639],
		   [ 0.        ,  1.5       , -4.89897949,  0.8660254 ,  4.24264069,  5.47722558],
		   [ 0.        ,  1.5       ,  1.22474487,  0.8660254 , -1.06066017, -1.36930639],
		   [ 0.        ,  1.5       ,  1.22474487,  0.8660254 ,  4.24264069,  2.73861279],
		   [ 0.        ,  1.5       ,  7.34846923,  0.8660254 ,  4.24264069,  0.        ]])
	"""
	V2Dr = np.zeros((r.size, int((degree+1)*(degree+2)/2)), dtype = np.float64)
	V2Ds = np.zeros((r.size, int((degree+1)*(degree+2)/2)), dtype = np.float64)

	a, b = rs2ab(r,s)

	sk = 0
	for i in range(0,degree+1):
		for j in range(0,degree+1-i):
			V2Dr[:,sk], V2Ds[:,sk] = GradSimplex2DP(a,b,i,j)
			sk += 1

	return (V2Dr, V2Ds)

def Dmatrices2D(degree,r,s,V):
	"""
	Initialize the (r,s) differentiation matrices on the simplex, evaluated at (r,s) at order N

	Parameters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle
		- ``V`` (``float64 array``) : Vandermonde matrix in 2D

	Returns
		- ``Dr`` (``float64 array``) : differentiation matrix along r-direction
		- ``Ds`` (``float64 array``) : differentiation matrix along s-direction

	Example
		>>> N = 2
		>>> r, s = RefNodes_Tri(N)
		>>> V = Vandermonde2D(N,r,s)
		>>> Dr, Ds = Dmatrices2D(N,r,s,V)
		>>> Dr
		array([[-1.5,  2. , -0.5,  0. ,  0. ,  0. ],
		   [-0.5,  0. ,  0.5,  0. ,  0. ,  0. ],
		   [ 0.5, -2. ,  1.5,  0. ,  0. ,  0. ],
		   [-0.5,  1. , -0.5, -1. ,  1. ,  0. ],
		   [ 0.5, -1. ,  0.5, -1. ,  1. ,  0. ],
		   [ 0.5,  0. , -0.5, -2. ,  2. ,  0. ]])
		>>> Ds
		array([[ -1.50000000e+00,   2.22044605e-16,   2.22044605e-16,   2.00000000e+00,  -4.44089210e-16,  -5.00000000e-01],
		   [ -5.00000000e-01,  -1.00000000e+00,   2.77555756e-17,   1.00000000e+00,   1.00000000e+00,  -5.00000000e-01],
		   [  5.00000000e-01,  -2.00000000e+00,  -2.22044605e-16,  -3.99042031e-16,   2.00000000e+00,  -5.00000000e-01],
		   [ -5.00000000e-01,   1.38777878e-16,   1.38777878e-16,   4.42493564e-17,  -2.22044605e-16,   5.00000000e-01],
		   [  5.00000000e-01,  -1.00000000e+00,   1.38777878e-17,  -1.00000000e+00,   1.00000000e+00,   5.00000000e-01],
		   [  5.00000000e-01,   1.11022302e-16,   1.66533454e-16,  -2.00000000e+00,   0.00000000e+00,   1.50000000e+00]])
	"""
	Vr, Vs = GradVandermonde2D(degree, r, s)
	invV = np.linalg.inv(V)
	Dr = np.dot(Vr,invV)
	Ds = np.dot(Vs,invV)

	return (Dr, Ds)

def RefNodes_Rect(degree):
	"""
	Compute (r,s) nodes in reference rectangle for polynomial of order `degree`

	Parameters
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``r`` (``float64 array``) : x-coordinates of nodes in the reference rectangle
		- ``s`` (``float64 array``) : y-coordinates of nodes in the reference rectangle

	Example
		>>> N = 3
		>>> r, s = RefNodes_Rect(N)
	"""
	LGLpts = nJacobiGL(0,0,degree)
	r = np.matrix(LGLpts).T * np.ones((1,degree+1))
	s = r
	r = r.T
	r = r.flatten()
	s = s.flatten()
	return (r,s)

def Simplex2DP_Rect(a,b,i,j):
	"""
	Compute 2D orthonormal polynomial on simplex(rectangle) at (a, b) of order (i, j)

	Parameters
		- ``a`` (``folat64 array``) : the value for the first normalized Jacobi polynomial in modal basis
		- ``b`` (``float64 array``) : the value for the second normalized Jacobi polynomial in modal basis
		- ``i`` (``int32``) : order of the the first normalized Jacobi polynomial in modal basis
		- ``j`` (``int32``) : order of the the second normalized Jacobi polynomial in modal basis

	Returns
		- ``P`` (``float64 array``) : evaluated value
	"""
	h1 = nJacobiP(a,0,0,i)
	h2 = nJacobiP(b,0,0,j)
	P = h1 * h2
	return P

def Vandermonde2D_Rect(degree,r,s):
	"""
	Initialize the 2D Vandermonde Matrix, :math:`V_{i,j} = \\phi_j(r_i)`

	Parameters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference rectangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference rectangle

	Returns
		- ``V2D`` (``float64 array``) : Vandermonde matrix in 2D
	"""
	V2D = np.zeros((r.size, (degree + 1) * (degree + 1)), dtype = np.float64)
	sk = 0
	for i in range(0, degree + 1):
		for j in range(0, degree + 1):
			V2D[:, sk] = Simplex2DP_Rect(r,s,i,j)
			sk = sk + 1
	return V2D

def GradSimplex2DP_Rect(r,s,id,jd):
	"""
	Return the derivatives of the modal basis (id,jd) on the rectangle at (a,b).

	Parameters
		- ``r`` (``float64``) : 
		- ``s`` (``float64``) : 
		- ``id`` (``int32``) : order of the the first normalized Jacobi polynomial in modal basis
		- ``jd`` (``int32``) : order of the the second normalized Jacobi polynomial in modal basis

	Returns
		- ``dmodedr`` (``float64 array``) : derivative value of modal basis on the rectangle along r-direction
		- ``dmodeds`` (``float64 array``) : derivative value of modal basis on the rectangle along s-direction
	"""
	fr = nJacobiP(r, 0, 0, id)
	dfr = DnJacobiP(r, 0, 0, id)
	gs = nJacobiP(s, 0, 0, jd)
	dgs = DnJacobiP(s, 0, 0, jd)

	dmodedr = dfr * gs
	dmodeds = fr * dgs
	return (dmodedr, dmodeds)


def GradVandermonde2D_Rect(degree,r,s):
	"""
	Initialize the gradient of the modal basis (i,j) at (r,s) at order N

	Parameters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle

	Returns
		- ``V2Dr`` (``float64 array``) : Gradient of Vandermonde matrix on the 2D simplex along r-direction
		- ``V2Ds`` (``float64 array``) : Gradient of Vandermonde matrix on the 2D simplex along s-direction
	"""
	V2Dr = np.zeros((r.size, (degree + 1) * (degree + 1)), dtype = np.float64)
	V2Ds = np.zeros((r.size, (degree + 1) * (degree + 1)), dtype = np.float64)

	sk = 0
	for i in range(0,degree+1):
		for j in range(0,degree+1):
			V2Dr[:,sk], V2Ds[:,sk] = GradSimplex2DP_Rect(r,s,i,j)
			sk += 1

	return (V2Dr, V2Ds)

def Dmatrices2D_Rect(degree,r,s,V):
	"""
	Initialize the (r,s) differentiation matrices on the simplex, evaluated at (r,s) at order N

	Parameters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference triangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference triangle
		- ``V`` (``float64 array``) : Vandermonde matrix in 2D

	Returns
		- ``Dr`` (``float64 array``) : differentiation matrix along r-direction
		- ``Ds`` (``float64 array``) : differentiation matrix along s-direction

	"""
	Vr, Vs = GradVandermonde2D_Rect(degree, r, s)
	invV = np.linalg.inv(V)
	Dr = np.dot(Vr,invV)
	Ds = np.dot(Vs,invV)

	return (Dr, Ds)

def RefNodes_Cube(degree):
	"""
	Compute (r,s) nodes in reference cube for polynomial of order `degree`

	Parameters
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``r`` (``float64 array``) : x-coordinates of nodes in the reference cube
		- ``s`` (``float64 array``) : y-coordinates of nodes in the reference cube
		- ``t`` (``float64 array``) : y-coordinates of nodes in the reference cube

	Example
		>>> N = 3
		>>> r, s, t = RefNodes_Cube(N)
	"""
	pts = np.linspace(-1,1,degree+1)
	import numpy.matlib
	r = np.matlib.repmat(pts,(degree+1)**2,1).flatten()
	s = np.matlib.repmat(np.matlib.repmat(pts,degree+1,1),1,degree+1).flatten('F')
	t = np.matlib.repmat(pts,(degree+1)**2,1).flatten('F')
	return (r,s,t)


def RefNodes_Rec(degree):
	"""
	Compute uniform (r,s) nodes in reference rectangle for polynomial of order `degree`

	Parameters
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``r`` (``float64 array``) : x-coordinates of nodes in the reference rectangle
		- ``s`` (``float64 array``) : y-coordinates of nodes in the reference rectangle

	Example
		>>> N = 3
		>>> r, s = RefNodes_Rec(N)
	"""
	pts = np.linspace(-1,1,degree+1)
	import numpy.matlib
	r = np.matlib.repmat(pts,degree+1,1).flatten()
	s = np.matlib.repmat(pts,degree+1,1).flatten('F')
	return (r,s)


def Simplex3DP_Cube(a,b,c,i,j,k):
	"""
	Compute 3D orthonormal polynomial on simplex(Cube) at (a, b) of order (i, j)

	Parameters
		- ``a`` (``folat64 array``) : the value for the first normalized Jacobi polynomial in modal basis
		- ``b`` (``float64 array``) : the value for the second normalized Jacobi polynomial in modal basis
		- ``c`` (``float64 array``) : the value for the third normalized Jacobi polynomial in modal basis
		- ``i`` (``int32``) : order of the the first normalized Jacobi polynomial in modal basis
		- ``j`` (``int32``) : order of the the second normalized Jacobi polynomial in modal basis
		- ``k`` (``int32``) : order of the the third normalized Jacobi polynomial in modal basis

	Returns
		- ``P`` (``float64 array``) : evaluated value
	"""
	h1 = nJacobiP(a,0,0,i)
	h2 = nJacobiP(b,0,0,j)
	h3 = nJacobiP(c,0,0,k)
	P = h1 * h2 * h3
	return P

def GradSimplex3DP_Cube(r,s,t,id,jd,kd):
	"""
	Return the derivatives of the modal basis (id,jd,kd) on the cube at (r,s,t).

	Parameters
		- ``r`` (``float64``) : x-coordinate on the cube
		- ``s`` (``float64``) : y-coordinate on the cube
		- ``t`` (``float64``) : z-coordinate on the cube
		- ``id`` (``int32``) : order of the the first normalized Jacobi polynomial in modal basis
		- ``jd`` (``int32``) : order of the the second normalized Jacobi polynomial in modal basis
		- ``id`` (``int32``) : order of the the third normalized Jacobi polynomial in modal basis

	Returns
		- ``dmodedr`` (``float64 array``) : derivative value of modal basis on the rectangle along r-direction
		- ``dmodeds`` (``float64 array``) : derivative value of modal basis on the rectangle along s-direction
		- ``dmodedt`` (``float64 array``) : derivative value of modal basis on the rectangle along t-direction
	"""
	fr = nJacobiP(r, 0, 0, id)
	dfr = DnJacobiP(r, 0, 0, id)
	gs = nJacobiP(s, 0, 0, jd)
	dgs = DnJacobiP(s, 0, 0, jd)
	ht = nJacobiP(t, 0, 0, kd)
	dht = DnJacobiP(t, 0, 0, kd)

	dmodedr = dfr * gs * ht
	dmodeds = fr * dgs * ht
	dmodedt = fr * gs * dht
	return (dmodedr, dmodeds, dmodedt)

def Vandermonde3D_Cube(degree,r,s,t):
	"""
	Initialize the 3D Vandermonde Matrix, :math:`V_{i,j} = \\phi_j(r_i)`

	Parameters
		- ``degree`` (``int32``) : Polynomial degree
		- ``r`` (``float64 array``) : x-coordinates of uniform nodes in the reference rectangle
		- ``s`` (``float64 array``) : y-coordinates of uniform nodes in the reference rectangle
		- ``t`` (``float64 array``) : z-coordinates of uniform nodes in the reference rectangle

	Returns
		- ``V3D`` (``float64 array``) : Vandermonde matrix in 3D
	"""
	V3D = np.zeros((r.size, (degree + 1) **3), dtype = np.float64)
	sk = 0
	for i in range(0, degree + 1):
		for j in range(0, degree + 1):
			for k in range(0, degree + 1):
				V3D[:, sk] = Simplex3DP_Cube(r,s,t,i,j,k)
				sk = sk + 1
	return V3D