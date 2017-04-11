from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import nJacobiP, DnJacobiP

def solve(c4n, ind4e, n4e, n4Db, f, u_D, degree):
	"""
	Computes the coordinates of nodes and elements.
	
	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``ind4e`` (``int32 array``) : indices for elements 
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4Db`` (``int32 array``) : nodes for Dirichlet boundary		
		- ``f`` (``lambda``) : source term 
		- ``u_D`` (``lambda``) : Dirichlet boundary condition
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : solution

	Example
	"""

	M_R, Srr_R, Sss_R, Dr_R, Ds_R = getMatrix(degree)
	fval = f(c4n[ind4e,0],c4n[ind4e,1]).flatten()
	nrNodes = int(c4n.shape[0])
	nrElems = int(n4e.shape[0])
	nrLocal = int(M_R.shape[0])

	I = np.zeros((nrElems * nrLocal * nrLocal), dtype = np.int32)
	J = np.zeros((nrElems * nrLocal * nrLocal), dtype = np.int32)
	
	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype = np.float64)
	b = np.zeros(nrNodes)
	Poisson_2D_Rectangle = lib['Poisson_2D_Rectangle']
	Poisson_2D_Rectangle.argtypes = (c_void_p, c_void_p, c_void_p, c_int,
	                    		  	 c_void_p, c_void_p, c_void_p, c_int,
	                    			 c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,)
	print(n4e)
	print(ind4e)
	Poisson_2D_Rectangle.restype = None
	Poisson_2D_Rectangle(c_void_p(n4e.ctypes.data), 
						 c_void_p(ind4e.ctypes.data),
					     c_void_p(c4n.ctypes.data), 
					     c_int(nrElems),
					     c_void_p(M_R.ctypes.data),
					     c_void_p(Srr_R.ctypes.data),
					     c_void_p(Sss_R.ctypes.data),
					     c_int(nrLocal),
					     c_void_p(fval.ctypes.data),
					     c_void_p(I.ctypes.data),
					     c_void_p(J.ctypes.data),
					     c_void_p(Alocal.ctypes.data),
					     c_void_p(b.ctypes.data))

	from scipy.sparse import coo_matrix
	from scipy.sparse.linalg import spsolve
	STIMA_COO = coo_matrix((Alocal, (I, J)), shape = (nrNodes, nrNodes))
	STIMA_CSR = STIMA_COO.tocsr()

	dof = np.setdiff1d(range(0,nrNodes), n4Db)

	x = np.zeros(nrNodes)
	x[dof] = spsolve(STIMA_CSR[dof,:].tocsc()[:, dof].tocsr(), b[dof])
	return x


def getMatrix(degree):
	"""
	Get FEM matrices on the reference domain I = [-1, 1]x[-1, 1]

	Paramters
		- ``degree`` (``int32``) : degree of polynomial

	Returns
		- ``M_R`` (``float64 array``) : Mass matrix on the reference domain
		- ``Srr_R`` (``float64 array``) : Stiffness matrix w.r.t. rr on the reference domain
		- ``Sss_R`` (``float64 array``) : Stiffness matrix w.r.t. ss on the reference domain
		- ``Dr_R`` (``float64 array``) : Differentiation matrix w.r.t. r on the reference domain  
		- ``Ds_R`` (``float64 array``) : Differentiation matrix w.r.t. s on the reference domain
	"""

	r = np.linspace(-1, 1, degree+1)
	V = VandermondeM1D(degree, r)
	invV = np.linalg.inv(V)
	M_R = np.kron(np.dot(np.transpose(invV),invV),np.dot(np.transpose(invV),invV))
	Dr_R = np.kron(np.eye(r.size),Dmatrix1D(degree, r, V))
	Ds_R = np.kron(Dmatrix1D(degree, r, V),np.eye(r.size))
	Srr_R = np.dot(np.dot(np.transpose(Dr_R),M_R),Dr_R)
	Sss_R = np.dot(np.dot(np.transpose(Ds_R),M_R),Ds_R)
	return (M_R, Srr_R, Sss_R, Dr_R, Ds_R)

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