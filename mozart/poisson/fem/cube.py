from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import nJacobiP, DnJacobiP, VandermondeM1D, Dmatrix1D

def compute_n4s(n4e):
	"""
	Get a matrix whose each row contains end points of the corresponding side (or edge)

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``n4s`` (``int32 array``) : nodes for sides

	Example
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> n4s = compute_n4s(n4e)
		>>> n4s
		array([[ 0,  1],
		   [ 1,  2],
		   [ 1,  4],
		   [ 2,  5],
		   [ 4,  3],
		   [ 5,  4],
		   [ 3,  0],
		   [ 0,  6],
		   [ 1,  7],
		   [ 2,  8],
		   [ 4, 10],
		   [ 5, 11],
		   [ 3,  9],
		   [ 6,  7],
		   [ 7,  8],
		   [ 7, 10],
		   [ 8, 11],
		   [10,  9],
		   [11, 10],
		   [ 9,  6]])
	"""
	allSides = np.array([n4e[:,[0,1]], n4e[:,[1,2]], n4e[:,[2,3]], n4e[:,[3,0]],
		n4e[:,[0,4]], n4e[:,[1,5]], n4e[:,[2,6]], n4e[:,[3,7]],
		n4e[:,[4,5]], n4e[:,[5,6]], n4e[:,[6,7]], n4e[:,[7,4]]])
	allSides = np.reshape(allSides,(12*n4e.shape[0],-1))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind = np.unique(x + y*1.0j, return_index=True)
	n4sInd = np.sort(ind)
	n4s = allSides[n4sInd,:]
	return n4s

def compute_s4e(n4e):
	"""
	Get a matrix whose each row contains three side numbers of the corresponding element

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``s4e`` (``int32 array``) : sides for elements

	Example
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> s4e = compute_s4e(n4e)
		>>> s4e
		array([[ 0,  2,  4,  6,  7,  8, 10, 12, 13, 15, 17, 19],
		   [ 1,  3,  5,  2,  8,  9, 11, 10, 14, 16, 18, 15]])
	"""
	allSides = np.array([n4e[:,[0,1]], n4e[:,[1,2]], n4e[:,[2,3]], n4e[:,[3,0]],
		n4e[:,[0,4]], n4e[:,[1,5]], n4e[:,[2,6]], n4e[:,[3,7]],
		n4e[:,[4,5]], n4e[:,[5,6]], n4e[:,[6,7]], n4e[:,[7,4]]])
	allSides = np.reshape(allSides,(12*n4e.shape[0],-1))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	s4e = sideNr[back].reshape(12,-1).transpose().astype('int')
	return s4e

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
		>>> from mozart.mesh.rectangle import cube
		>>> c4n, ind4e, n4e, n4Db = cube(0,1,0,1,0,1,2,2,2,1)
		>>> f = lambda x,y,z: 3.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		>>> u_D = lambda x,y,z: 0*x
		>>> from mozart.poisson.fem.cube import solve
		>>> x = solve(c4n, ind4e, n4e, n4Db, f, u_D, 1)
		>>> x
		array([ 0.          0.          0.          0.          0.          0.
		        0.			0.          0.          0.          0.          0.
		        0.			0.82246703  0.          0.          0.          0.
		        0.          0.		    0.          0.          0.          0.
		        0.          0.          0.        ])
	"""
	M_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R = getMatrix(degree)
	fval = f(c4n[ind4e,0], c4n[ind4e,1], c4n[ind4e,2]).flatten()
	nrNodes = int(c4n.shape[0])
	nrElems = int(n4e.shape[0])
	nrLocal = int(M_R.shape[0])

	I = np.zeros((nrElems * nrLocal * nrLocal), dtype = np.int32)
	J = np.zeros((nrElems * nrLocal * nrLocal), dtype = np.int32)

	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype = np.float64)
	b = np.zeros(nrNodes)
	Poisson_3D_Cube = lib['Poisson_3D_Cube']
	Poisson_3D_Cube.argtypes = (c_void_p, c_void_p, c_void_p, c_int,
									 c_void_p, c_void_p, c_void_p, c_void_p, c_int,
									 c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,)

	c4n = c4n.flatten()
	ind4e = ind4e.flatten()
	n4e = n4e.flatten()

	Poisson_3D_Cube.restype = None
	Poisson_3D_Cube(c_void_p(n4e.ctypes.data),
						 c_void_p(ind4e.ctypes.data),
						 c_void_p(c4n.ctypes.data),
						 c_int(nrElems),
						 c_void_p(M_R.ctypes.data),
						 c_void_p(Srr_R.ctypes.data),
						 c_void_p(Sss_R.ctypes.data),
						 c_void_p(Stt_R.ctypes.data),
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

def computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, exact_uz, approx_u, degree, degree_i):
	"""
	Compute semi H^1-error between exact solution and approximate solution.

	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``ind4e`` (``int32 array``) : indices for elements
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``exact_u`` (``lambda``) : exact solution
		- ``exact_ux`` (``lambda``) : derivative of exact solution
		- ``exact_uy`` (``lambda``) : derivative of exact solution
		- ``exact_uz`` (``lambda``) : derivative of exact solution
		- ``approx_u`` (``float64 array``) : approximate solution
		- ``degree`` (``int32``) : polynomial degree
		- ``degree_i`` (``int32``) : polynomial degree for interpolation

	Returns
		- ``sH1error`` (``float64``) : semi H^1 error between exact solution and approximate solution.

	Example
		>>> from mozart.mesh.cube import cube
		>>> c4n, ind4e, n4e, n4Db = cube(0,1,0,1,0,1,4,4,4,1)
		>>> f = lambda x,y,z: 3.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		>>> u_D = lambda x,y,z: 0*x
		>>> from mozart.poisson.fem.cube import solve
		>>> x = solve(c4n, ind4e, n4e, n4Db, f, u_D, 1)
		>>> from mozart.poisson.fem.cube import computeError
		>>> exact_u = lambda x,y,z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		>>> exact_ux = lambda x,y,z: np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		>>> exact_uy = lambda x,y,z: np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
		>>> exact_uz = lambda x,y,z: np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)
		>>> sH1error = computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, exact_uz, approx_u, 1, 3)
		>>> sH1error
		0.38357333319
	"""
	# L2error = 0
	sH1error = 0

	M_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R = getMatrix(degree)

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
		xr = (c4n[n4e[j,1],0] - c4n[n4e[j,0],0]) / 2.0
		ys = (c4n[n4e[j,3],1] - c4n[n4e[j,0],1]) / 2.0
		zt = (c4n[n4e[j,4],2] - c4n[n4e[j,0],2]) / 2.0
		Jacobi = xr * ys * zt
		rx = 1.0 / xr
		sy = 1.0 / ys
		tz = 1.0 / zt

		Dex = exact_ux(c4n[ind4e[j],0],c4n[ind4e[j],1],c4n[ind4e[j],2]) - rx*np.dot(Dr_R,approx_u[ind4e[j]])
		Dey = exact_uy(c4n[ind4e[j],0],c4n[ind4e[j],1],c4n[ind4e[j],2]) - sy*np.dot(Ds_R,approx_u[ind4e[j]])
		Dez = exact_uz(c4n[ind4e[j],0],c4n[ind4e[j],1],c4n[ind4e[j],2]) - tz*np.dot(Dt_R,approx_u[ind4e[j]])
		sH1error += Jacobi*(np.dot(np.dot(np.transpose(Dex),M_R),Dex) + \
							np.dot(np.dot(np.transpose(Dey),M_R),Dey) + \
							np.dot(np.dot(np.transpose(Dez),M_R),Dez))

	sH1error = np.sqrt(sH1error)
	return sH1error

def getMatrix(degree):
	r = np.linspace(-1, 1, degree+1)
	V = VandermondeM1D(degree, r)
	invV = np.linalg.inv(V)
	M1_R = np.dot(np.transpose(invV),invV)

	M_R = np.kron(np.kron(M1_R,M1_R),M1_R)

	Dr_R = np.kron(np.eye(r.size),np.kron(np.eye(r.size),Dmatrix1D(degree, r, V)))
	Ds_R = np.kron(np.eye(r.size),np.kron(Dmatrix1D(degree, r, V),np.eye(r.size)))
	Dt_R = np.kron(Dmatrix1D(degree, r, V),np.kron(np.eye(r.size),np.eye(r.size)))

	Srr_R = np.dot(np.dot(np.transpose(Dr_R),M_R),Dr_R)
	Sss_R = np.dot(np.dot(np.transpose(Ds_R),M_R),Ds_R)
	Stt_R = np.dot(np.dot(np.transpose(Dt_R),M_R),Dt_R)

	return (M_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R)