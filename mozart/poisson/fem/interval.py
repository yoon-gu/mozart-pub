from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import nJacobiP, DnJacobiP, VandermondeM1D, DVandermondeM1D, Dmatrix1D

def solve(c4n,n4e,n4db,ind4e,f,u_D,degree):
	"""
	Computes the coordinates of nodes and elements.
	
	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4db`` (``int32 array``) : nodes for Dirichlet boundary
		- ``ind4e`` (``int32 array``) : indices for elements 
		- ``f`` (``lambda``) : source term 
		- ``u_D`` (``lambda``) : Dirichlet boundary condition
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : solution

	Example
		>>> N = 2
		>>> from mozart.mesh.rectangle import interval 
		>>> c4n, n4e, n4db, ind4e = interval(0, 1, 4, 2)
		>>> f = lambda x: np.ones_like(x)
		>>> u_D = lambda x: np.zeros_like(x)
		>>> from mozart.poisson.fem.interval import solve
		>>> x = solve(c4n, n4e, n4db, ind4e, f, u_D, N)
		>>> x
		array([ 0.       ,  0.0546875,  0.09375  ,  0.1171875,  0.125    ,
		   0.1171875,  0.09375  ,  0.0546875,  0.       ])
	"""
	M_R, S_R, D_R = getMatrix(degree)
	fval = f(c4n[ind4e].flatten())
	nrNodes = int(c4n.shape[0])
	nrElems = int(n4e.shape[0])
	nrLocal = int(M_R.shape[0])

	I = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	J = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.float64)

	b = np.zeros(nrNodes)
	Poison_1D = lib['Poisson_1D'] # need the extern!!
	Poison_1D.argtypes = (c_void_p, c_void_p, c_void_p, c_int,
	                    c_void_p, c_void_p, c_int,
	                    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,)
	Poison_1D.restype = None
	Poison_1D(c_void_p(n4e.ctypes.data), c_void_p(ind4e.ctypes.data),
	    c_void_p(c4n.ctypes.data), c_int(nrElems),
	    c_void_p(M_R.ctypes.data),
	    c_void_p(S_R.ctypes.data),
	    c_int(nrLocal),
	    c_void_p(fval.ctypes.data),
	    c_void_p(I.ctypes.data),
	    c_void_p(J.ctypes.data),
	    c_void_p(Alocal.ctypes.data),
	    c_void_p(b.ctypes.data))

	from scipy.sparse import coo_matrix
	from scipy.sparse.linalg import spsolve
	STIMA_COO = coo_matrix((Alocal, (I, J)), shape=(nrNodes, nrNodes))
	STIMA_CSR = STIMA_COO.tocsr()

	dof = np.setdiff1d(range(0,nrNodes), n4db)

	x = np.zeros(nrNodes)
	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return x


def computeError(c4n, n4e, ind4e, exact_u, exact_ux, approx_u, degree, degree_i):
	"""
	Computes L^2-error and semi H^1-error between exact solution and approximate solution.
	
	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``ind4e`` (``int32 array``) : indices for elements
		- ``exact_u`` (``lambda``) : exact solution
		- ``exact_ux`` (``lambda``) : derivative of exact solution 
		- ``approx_u`` (``float64 array``) : approximate solution
		- ``degree`` (``int32``) : Polynomial degree
		- ``degree_i`` (``int32``) : Polynomial degree for interpolation

	Returns
		- ``L2error`` (``float64``) : L^2 error between exact solution and approximate solution.
		- ``sH1error`` (``float64``) : semi H^1 error between exact solution and approximate solution.

	Example
		>>> N = 2
		>>> from mozart.mesh.rectangle import interval 
		>>> c4n, n4e, n4db, ind4e = interval(0, 1, 4, 2)
		>>> f = lambda x: np.pi ** 2 * np.sin(np.pi * x)
		>>> u_D = lambda x: np.zeros_like(x)
		>>> from mozart.poisson.fem.interval import solve_p
		>>> x = solve_p(c4n, n4e, n4db, ind4e, f, u_D, N)
		>>> from mozart.poisson.fem.interval import computeError
		>>> exact_u = lambda x: np.sin(np.pi * x)
		>>> exact_ux = lambda x: np.pi * np.cos(np.pi * x)
		>>> L2error, sH1error = computeError(c4n, n4e, ind4e, exact_u, exact_ux, x, N, N+3)
		>>> L2error
		0.0020225729623142077
		>>> sH1error
		0.05062779815975444
	"""
	L2error = 0
	sH1error = 0

	r = np.linspace(-1, 1, degree + 1)
	V = VandermondeM1D(degree, r)
	Dr = Dmatrix1D(degree, r, V)

	r_i = np.linspace(-1, 1, degree_i + 1)
	V_i = VandermondeM1D(degree_i, r_i)
	invV_i = np.linalg.inv(V_i)
	M_R = np.dot(np.transpose(invV_i), invV_i)
	PM = VandermondeM1D(degree, r_i)
	interpM = np.transpose(np.linalg.solve(np.transpose(V), np.transpose(PM)))

	for j in range(0,n4e.shape[0]):
		Jacobi = (c4n[n4e[j,1]] - c4n[n4e[j,0]])/2.0
		approx_u_i = np.dot(interpM, approx_u[ind4e[j]])
		Dapprox_u = np.dot(Dr, approx_u[ind4e[j]]) / Jacobi
		Dapprox_u_i = np.dot(interpM, Dapprox_u)

		nodes = (1-r_i)/2*c4n[n4e[j,0]]+(1+r_i)/2*c4n[n4e[j,1]]
		diff_u = exact_u(nodes) - approx_u_i
		diff_Du = exact_ux(nodes) - Dapprox_u_i
		L2error += Jacobi*np.dot(np.dot(np.transpose(diff_u),M_R),diff_u)
		sH1error += Jacobi*np.dot(np.dot(np.transpose(diff_Du),M_R),diff_Du)

	L2error = np.sqrt(L2error)
	sH1error = np.sqrt(sH1error)
	return (L2error, sH1error)

def getMatrix(degree):
	"""
	Get FEM matrices on the reference domain I = [-1, 1]

	Paramters
		- ``degree`` (``int32``) : degree of polynomial

	Returns
		- ``M_R`` (``float64 array``) : Mass matrix on the reference domain
		- ``S_R`` (``float64 array``) : Stiffness matrix on the reference domain
		- ``D_R`` (``float64 array``) : Differentiation matrix on the reference domain

	"""

	r = np.linspace(-1, 1, degree+1)
	V = VandermondeM1D(degree, r)
	invV = np.linalg.inv(V)
	M_R = np.dot(np.transpose(invV),invV)
	D_R = Dmatrix1D(degree, r, V)
	S_R = np.dot(np.dot(np.transpose(D_R),M_R),D_R)
	return (M_R, S_R, D_R)

