from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import nJacobiP, DnJacobiP, VandermondeM1D, Dmatrix1D

def solve(c4n, ind4e, n4e, n4Db, f, u_D, degree):
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