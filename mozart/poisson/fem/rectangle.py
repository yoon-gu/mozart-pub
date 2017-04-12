from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import nJacobiP, DnJacobiP, VandermondeM1D, Dmatrix1D, DVandermondeM1D

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
		>>> from mozart.mesh.rectangle import rectangle
		>>> c4n, ind4e, n4e, n4Db = rectangle(0,1,0,1,4,4,1)
		>>> f = lambda x,y: 2.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
		>>> u_D = lambda x,y: 0*x
		>>> from mozart.poisson.fem.rectangle import solve
		>>> x = solve(c4n, ind4e, n4e, n4Db, f, u_D, 1)
		>>> x
		array([ 0.          ,0.          ,0.          ,0.          ,0.
				0.			,0.47511045  ,0.67190765  ,0.47511045  ,0.
				0.          ,0.67190765  ,0.95022091  ,0.67190765  ,0.
				0.          ,0.47511045  ,0.67190765  ,0.47511045  ,0.
				0.          ,0.          ,0.          ,0.          ,0.])
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

	c4n = c4n.flatten()
	ind4e = ind4e.flatten()
	n4e = n4e.flatten()

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

def computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, approx_u, degree, degree_i):
	"""
	Compute semi H^1-error between exact solution and approximate solution.

	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``ind4e`` (``int32 array``) : indices for elements
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``exact_u`` (``lambda``) : exact solution
		- ``exact_ux`` (``lambda``) : derivative of exact solution
		- ``exact_uy`` (``lambda``) : derivative of exact solution
		- ``approx_u`` (``float64 array``) : approximate solution
		- ``degree`` (``int32``) : polynomial degree
		- ``degree_i`` (``int32``) : polynomial degree for interpolation

	Returns
		- ``sH1error`` (``float64``) : semi H^1 error between exact solution and approximate solution.

	Example
		>>> from mozart.mesh.rectangle import rectangle
		>>> c4n, ind4e, n4e, n4Db = rectangle(0,1,0,1,4,4,1)
		>>> f = lambda x,y: 2.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
		>>> u_D = lambda x,y: 0*x
		>>> from mozart.poisson.fem.rectangle import solve
		>>> x = solve(c4n, ind4e, n4e, n4Db, f, u_D, 1)
		>>> from mozart.poisson.fem.rectangle import computeError
		>>> exact_u = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
		>>> exact_ux = lambda x,y: np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
		>>> exact_uy = lambda x,y: np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
		>>> sH1error = computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, approx_u, 1, 3)
		>>> sH1error
		0.466257369082
	"""
	# L2error = 0
	sH1error = 0

	M_R, Srr_R, Sss_R, Dr_R, Ds_R = getMatrix(degree)

	# r = np.linspace(-1, 1, degree + 1)
	# V = VandermondeM1D(degree, r)
	# Dr = Dmatrix1D(degree, r, V)

	# r_i = np.linspace(-1, 1, degree_i + 1)
	# V_i = VandermondeM1D(degree_i, r_i)
	# invV_i = np.linalg.inv(V_i)
	# M_R = np.dot(np.transpose(invV_i), invV_i)
	# PM = VandermondeM1D(degree, r_i)
	# interpM = np.transpose(np.linalg.solve(np.transpose(V), np.transpose(PM)))


	for j in range(0,n4e.shape[0]):
		xr = (c4n[n4e[j,1],0] - c4n[n4e[j,0],0])/2.0
		ys = (c4n[n4e[j,3],1] - c4n[n4e[j,0],1])/2.0
		Jacobi = xr*ys
		rx = ys/Jacobi
		sy = xr/Jacobi

		Dex = exact_ux(c4n[ind4e[j],0],c4n[ind4e[j],1]) - rx*np.dot(Dr_R,approx_u[ind4e[j]])
		Dey = exact_uy(c4n[ind4e[j],0],c4n[ind4e[j],1]) - sy*np.dot(Ds_R,approx_u[ind4e[j]])
		sH1error += Jacobi*(np.dot(np.dot(np.transpose(Dex),M_R),Dex) + np.dot(np.dot(np.transpose(Dey),M_R),Dey))

	sH1error = np.sqrt(sH1error)

	return sH1error

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