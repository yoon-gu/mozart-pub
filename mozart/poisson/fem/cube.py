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
	if degree == 1:
		M_R = np.array([[8, 4, 4, 2, 4, 2, 2, 1],
						[4, 8, 2, 4, 2, 4, 1, 2],
						[4, 2, 8, 4, 2, 1, 4, 2],
						[2, 4, 4, 8, 1, 2, 2, 4],
						[4, 2, 2, 1, 8, 4, 4, 2],
						[2, 4, 1, 2, 4, 8, 2, 4],
						[2, 1, 4, 2, 4, 2, 8, 4],
						[1, 2, 2, 4, 2, 4, 4, 8]]) / 27.

		Srr_R = np.array([[ 4, -4,  2, -2,  2, -2,  1, -1],
						[-4,  4, -2,  2, -2,  2, -1,  1],
						[ 2, -2,  4, -4,  1, -1,  2, -2],
						[-2,  2, -4,  4, -1,  1, -2,  2],
						[ 2, -2,  1, -1,  4, -4,  2, -2],
						[-2,  2, -1,  1, -4,  4, -2,  2],
						[ 1, -1,  2, -2,  2, -2,  4, -4],
						[-1,  1, -2,  2, -2,  2, -4,  4]]) / 18.

		Sss_R = np.array([[ 4,  2, -4, -2,  2,  1, -2, -1],
			          [ 2,  4, -2, -4,  1,  2, -1, -2],
			          [-4, -2,  4,  2, -2, -1,  2,  1],
			          [-2, -4,  2,  4, -1, -2,  1,  2],
			          [ 2,  1, -2, -1,  4,  2, -4, -2],
			          [ 1,  2, -1, -2,  2,  4, -2, -4],
			          [-2, -1,  2,  1, -4, -2,  4,  2],
			          [-1, -2,  1,  2, -2, -4,  2,  4]]) / 18.
		Stt_R = np.array([[ 4,  2,  2,  1, -4, -2, -2, -1],
			          [ 2,  4,  1,  2, -2, -4, -1, -2],
			          [ 2,  1,  4,  2, -2, -1, -4, -2],
			          [ 1,  2,  2,  4, -1, -2, -2, -4],
			          [-4, -2, -2, -1,  4,  2,  2,  1],
			          [-2, -4, -1, -2,  2,  4,  1,  2],
			          [-2, -1, -4, -2,  2,  1,  4,  2],
			          [-1, -2, -2, -4,  1,  2,  2,  4]]) / 18.
		Dr_R = np.array([[-1, 1,  0, 0,  0, 0,  0, 0],
			         [-1, 1,  0, 0,  0, 0,  0, 0],
			         [ 0, 0, -1, 1,  0, 0,  0, 0],
			         [ 0, 0, -1, 1,  0, 0,  0, 0],
			         [ 0, 0,  0, 0, -1, 1,  0, 0],
			         [ 0, 0,  0, 0, -1, 1,  0, 0],
			         [ 0, 0,  0, 0,  0, 0, -1, 1],
			         [ 0, 0,  0, 0,  0, 0, -1, 1]]) / 2.

		Ds_R = np.array([[-1,  0, 1, 0,  0,  0, 0, 0],
			         [ 0, -1, 0, 1,  0,  0, 0, 0],
			         [-1,  0, 1, 0,  0,  0, 0, 0],
			         [ 0, -1, 0, 1,  0,  0, 0, 0],
			         [ 0,  0, 0, 0, -1,  0, 1, 0],
			         [ 0,  0, 0, 0,  0, -1, 0, 1],
			         [ 0,  0, 0, 0, -1,  0, 1, 0],
			         [ 0,  0, 0, 0,  0, -1, 0, 1]]) / 2.

		Dt_R = np.array([[-1,  0,  0,  0, 1, 0, 0, 0],
			         [ 0, -1,  0,  0, 0, 1, 0, 0],
			         [ 0,  0, -1,  0, 0, 0, 1, 0],
			         [ 0,  0,  0, -1, 0, 0, 0, 1],
			         [-1,  0,  0,  0, 1, 0, 0, 0],
			         [ 0, -1,  0,  0, 0, 1, 0, 0],
			         [ 0,  0, -1,  0, 0, 0, 1, 0],
			         [ 0,  0,  0, -1, 0, 0, 0, 1]]) / 2.

	elif degree == 2:
		 M_R = np.array([ [ 64,  32, -16,  32,   16,  -8, -16,  -8,   4,  32,   16,  -8,   16,    8,   -4,  -8,   -4,   2, -16,  -8,   4,  -8,   -4,   2,   4,   2,  -1],
		 		         [ 32, 256,  32,  16,  128,  16,  -8, -64,  -8,  16,  128,  16,    8,   64,    8,  -4,  -32,  -4,  -8, -64,  -8,  -4,  -32,  -4,   2,  16,   2],
		 		         [-16,  32,  64,  -8,   16,  32,   4,  -8, -16,  -8,   16,  32,   -4,    8,   16,   2,   -4,  -8,   4,  -8, -16,   2,   -4,  -8,  -1,   2,   4],
		 		         [ 32,  16,  -8, 256,  128, -64,  32,  16,  -8,  16,    8,  -4,  128,   64,  -32,  16,    8,  -4,  -8,  -4,   2, -64,  -32,  16,  -8,  -4,   2],
		 		         [ 16, 128,  16, 128, 1024, 128,  16, 128,  16,   8,   64,   8,   64,  512,   64,   8,   64,   8,  -4, -32,  -4, -32, -256, -32,  -4, -32,  -4],
		 		         [ -8,  16,  32, -64,  128, 256,  -8,  16,  32,  -4,    8,  16,  -32,   64,  128,  -4,    8,  16,   2,  -4,  -8,  16,  -32, -64,   2,  -4,  -8],
		 		         [-16,  -8,   4,  32,   16,  -8,  64,  32, -16,  -8,   -4,   2,   16,    8,   -4,  32,   16,  -8,   4,   2,  -1,  -8,   -4,   2, -16,  -8,   4],
		 		         [ -8, -64,  -8,  16,  128,  16,  32, 256,  32,  -4,  -32,  -4,    8,   64,    8,  16,  128,  16,   2,  16,   2,  -4,  -32,  -4,  -8, -64,  -8],
		 		         [  4,  -8, -16,  -8,   16,  32, -16,  32,  64,   2,   -4,  -8,   -4,    8,   16,  -8,   16,  32,  -1,   2,   4,   2,   -4,  -8,   4,  -8, -16],
		 		         [ 32,  16,  -8,  16,    8,  -4,  -8,  -4,   2, 256,  128, -64,  128,   64,  -32, -64,  -32,  16,  32,  16,  -8,  16,    8,  -4,  -8,  -4,   2],
		 		         [ 16, 128,  16,   8,   64,   8,  -4, -32,  -4, 128, 1024, 128,   64,  512,   64, -32, -256, -32,  16, 128,  16,   8,   64,   8,  -4, -32,  -4],
		 		         [ -8,  16,  32,  -4,    8,  16,   2,  -4,  -8, -64,  128, 256,  -32,   64,  128,  16,  -32, -64,  -8,  16,  32,  -4,    8,  16,   2,  -4,  -8],
		 		         [ 16,   8,  -4, 128,   64, -32,  16,   8,  -4, 128,   64, -32, 1024,  512, -256, 128,   64, -32,  16,   8,  -4, 128,   64, -32,  16,   8,  -4],
		 		         [  8,  64,   8,  64,  512,  64,   8,  64,   8,  64,  512,  64,  512, 4096,  512,  64,  512,  64,   8,  64,   8,  64,  512,  64,   8,  64,   8],
		 		         [ -4,   8,  16, -32,   64, 128,  -4,   8,  16, -32,   64, 128, -256,  512, 1024, -32,   64, 128,  -4,   8,  16, -32,   64, 128,  -4,   8,  16],
		 		         [ -8,  -4,   2,  16,    8,  -4,  32,  16,  -8, -64,  -32,  16,  128,   64,  -32, 256,  128, -64,  -8,  -4,   2,  16,    8,  -4,  32,  16,  -8],
		 		         [ -4, -32,  -4,   8,   64,   8,  16, 128,  16, -32, -256, -32,   64,  512,   64, 128, 1024, 128,  -4, -32,  -4,   8,   64,   8,  16, 128,  16],
		 		         [  2,  -4,  -8,  -4,    8,  16,  -8,  16,  32,  16,  -32, -64,  -32,   64,  128, -64,  128, 256,   2,  -4,  -8,  -4,    8,  16,  -8,  16,  32],
		 		         [-16,  -8,   4,  -8,   -4,   2,   4,   2,  -1,  32,   16,  -8,   16,    8,   -4,  -8,   -4,   2,  64,  32, -16,  32,   16,  -8, -16,  -8,   4],
		 		         [ -8, -64,  -8,  -4,  -32,  -4,   2,  16,   2,  16,  128,  16,    8,   64,    8,  -4,  -32,  -4,  32, 256,  32,  16,  128,  16,  -8, -64,  -8],
		 		         [  4,  -8, -16,   2,   -4,  -8,  -1,   2,   4,  -8,   16,  32,   -4,    8,   16,   2,   -4,  -8, -16,  32,  64,  -8,   16,  32,   4,  -8, -16],
		 		         [ -8,  -4,   2, -64,  -32,  16,  -8,  -4,   2,  16,    8,  -4,  128,   64,  -32,  16,    8,  -4,  32,  16,  -8, 256,  128, -64,  32,  16,  -8],
		 		         [ -4, -32,  -4, -32, -256, -32,  -4, -32,  -4,   8,   64,   8,   64,  512,   64,   8,   64,   8,  16, 128,  16, 128, 1024, 128,  16, 128,  16],
		 		         [  2,  -4,  -8,  16,  -32, -64,   2,  -4,  -8,  -4,    8,  16,  -32,   64,  128,  -4,    8,  16,  -8,  16,  32, -64,  128, 256,  -8,  16,  32],
		 		         [  4,   2,  -1,  -8,   -4,   2, -16,  -8,   4,  -8,   -4,   2,   16,    8,   -4,  32,   16,  -8, -16,  -8,   4,  32,   16,  -8,  64,  32, -16],
		 		         [  2,  16,   2,  -4,  -32,  -4,  -8, -64,  -8,  -4,  -32,  -4,    8,   64,    8,  16,  128,  16,  -8, -64,  -8,  16,  128,  16,  32, 256,  32],
		 		         [ -1,   2,   4,   2,   -4,  -8,   4,  -8, -16,   2,   -4,  -8,   -4,    8,   16,  -8,   16,  32,   4,  -8, -16,  -8,   16,  32, -16,  32,  64]]) / 3375.

		 Srr_R = np.array([[ 112, -128,   16,   56,  -64,    8,  -28,   32,   -4,   56,  -64,    8,    28,   -32,     4,  -14,   16,   -2,  -28,   32,   -4,  -14,   16,   -2,    7,   -8,    1],
							[-128,  256, -128,  -64,  128,  -64,   32,  -64,   32,  -64,  128,  -64,   -32,    64,   -32,   16,  -32,   16,   32,  -64,   32,   16,  -32,   16,   -8,   16,   -8],
							[  16, -128,  112,    8,  -64,   56,   -4,   32,  -28,    8,  -64,   56,     4,   -32,    28,   -2,   16,  -14,   -4,   32,  -28,   -2,   16,  -14,    1,   -8,    7],
							[  56,  -64,    8,  448, -512,   64,   56,  -64,    8,   28,  -32,    4,   224,  -256,    32,   28,  -32,    4,  -14,   16,   -2, -112,  128,  -16,  -14,   16,   -2],
							[ -64,  128,  -64, -512, 1024, -512,  -64,  128,  -64,  -32,   64,  -32,  -256,   512,  -256,  -32,   64,  -32,   16,  -32,   16,  128, -256,  128,   16,  -32,   16],
							[   8,  -64,   56,   64, -512,  448,    8,  -64,   56,    4,  -32,   28,    32,  -256,   224,    4,  -32,   28,   -2,   16,  -14,  -16,  128, -112,   -2,   16,  -14],
							[ -28,   32,   -4,   56,  -64,    8,  112, -128,   16,  -14,   16,   -2,    28,   -32,     4,   56,  -64,    8,    7,   -8,    1,  -14,   16,   -2,  -28,   32,   -4],
							[  32,  -64,   32,  -64,  128,  -64, -128,  256, -128,   16,  -32,   16,   -32,    64,   -32,  -64,  128,  -64,   -8,   16,   -8,   16,  -32,   16,   32,  -64,   32],
							[  -4,   32,  -28,    8,  -64,   56,   16, -128,  112,   -2,   16,  -14,     4,   -32,    28,    8,  -64,   56,    1,   -8,    7,   -2,   16,  -14,   -4,   32,  -28],
							[  56,  -64,    8,   28,  -32,    4,  -14,   16,   -2,  448, -512,   64,   224,  -256,    32, -112,  128,  -16,   56,  -64,    8,   28,  -32,    4,  -14,   16,   -2],
							[ -64,  128,  -64,  -32,   64,  -32,   16,  -32,   16, -512, 1024, -512,  -256,   512,  -256,  128, -256,  128,  -64,  128,  -64,  -32,   64,  -32,   16,  -32,   16],
							[   8,  -64,   56,    4,  -32,   28,   -2,   16,  -14,   64, -512,  448,    32,  -256,   224,  -16,  128, -112,    8,  -64,   56,    4,  -32,   28,   -2,   16,  -14],
							[  28,  -32,    4,  224, -256,   32,   28,  -32,    4,  224, -256,   32,  1792, -2048,   256,  224, -256,   32,   28,  -32,    4,  224, -256,   32,   28,  -32,    4],
							[ -32,   64,  -32, -256,  512, -256,  -32,   64,  -32, -256,  512, -256, -2048,  4096, -2048, -256,  512, -256,  -32,   64,  -32, -256,  512, -256,  -32,   64,  -32],
							[   4,  -32,   28,   32, -256,  224,    4,  -32,   28,   32, -256,  224,   256, -2048,  1792,   32, -256,  224,    4,  -32,   28,   32, -256,  224,    4,  -32,   28],
							[ -14,   16,   -2,   28,  -32,    4,   56,  -64,    8, -112,  128,  -16,   224,  -256,    32,  448, -512,   64,  -14,   16,   -2,   28,  -32,    4,   56,  -64,    8],
							[  16,  -32,   16,  -32,   64,  -32,  -64,  128,  -64,  128, -256,  128,  -256,   512,  -256, -512, 1024, -512,   16,  -32,   16,  -32,   64,  -32,  -64,  128,  -64],
							[  -2,   16,  -14,    4,  -32,   28,    8,  -64,   56,  -16,  128, -112,    32,  -256,   224,   64, -512,  448,   -2,   16,  -14,    4,  -32,   28,    8,  -64,   56],
							[ -28,   32,   -4,  -14,   16,   -2,    7,   -8,    1,   56,  -64,    8,    28,   -32,     4,  -14,   16,   -2,  112, -128,   16,   56,  -64,    8,  -28,   32,   -4],
							[  32,  -64,   32,   16,  -32,   16,   -8,   16,   -8,  -64,  128,  -64,   -32,    64,   -32,   16,  -32,   16, -128,  256, -128,  -64,  128,  -64,   32,  -64,   32],
							[  -4,   32,  -28,   -2,   16,  -14,    1,   -8,    7,    8,  -64,   56,     4,   -32,    28,   -2,   16,  -14,   16, -128,  112,    8,  -64,   56,   -4,   32,  -28],
							[ -14,   16,   -2, -112,  128,  -16,  -14,   16,   -2,   28,  -32,    4,   224,  -256,    32,   28,  -32,    4,   56,  -64,    8,  448, -512,   64,   56,  -64,    8],
							[  16,  -32,   16,  128, -256,  128,   16,  -32,   16,  -32,   64,  -32,  -256,   512,  -256,  -32,   64,  -32,  -64,  128,  -64, -512, 1024, -512,  -64,  128,  -64],
							[  -2,   16,  -14,  -16,  128, -112,   -2,   16,  -14,    4,  -32,   28,    32,  -256,   224,    4,  -32,   28,    8,  -64,   56,   64, -512,  448,    8,  -64,   56],
							[   7,   -8,    1,  -14,   16,   -2,  -28,   32,   -4,  -14,   16,   -2,    28,   -32,     4,   56,  -64,    8,  -28,   32,   -4,   56,  -64,    8,  112, -128,   16],
							[  -8,   16,   -8,   16,  -32,   16,   32,  -64,   32,   16,  -32,   16,   -32,    64,   -32,  -64,  128,  -64,   32,  -64,   32,  -64,  128,  -64, -128,  256, -128],
							[   1,   -8,    7,   -2,   16,  -14,   -4,   32,  -28,   -2,   16,  -14,     4,   -32,    28,    8,  -64,   56,   -4,   32,  -28,    8,  -64,   56,   16, -128,  112]]) / 1350.

		 Sss_R = np.array([ [ 112,   56,  -28, -128,  -64,   32,   16,    8,   -4,   56,    28,  -14,  -64,   -32,   16,    8,     4,   -2,  -28,  -14,    7,   32,   16,   -8,   -4,   -2,    1],
		 		           [  56,  448,   56,  -64, -512,  -64,    8,   64,    8,   28,   224,   28,  -32,  -256,  -32,    4,    32,    4,  -14, -112,  -14,   16,  128,   16,   -2,  -16,   -2],
		 		           [ -28,   56,  112,   32,  -64, -128,   -4,    8,   16,  -14,    28,   56,   16,   -32,  -64,   -2,     4,    8,    7,  -14,  -28,   -8,   16,   32,    1,   -2,   -4],
		 		           [-128,  -64,   32,  256,  128,  -64, -128,  -64,   32,  -64,   -32,   16,  128,    64,  -32,  -64,   -32,   16,   32,   16,   -8,  -64,  -32,   16,   32,   16,   -8],
		 		           [ -64, -512,  -64,  128, 1024,  128,  -64, -512,  -64,  -32,  -256,  -32,   64,   512,   64,  -32,  -256,  -32,   16,  128,   16,  -32, -256,  -32,   16,  128,   16],
		 		           [  32,  -64, -128,  -64,  128,  256,   32,  -64, -128,   16,   -32,  -64,  -32,    64,  128,   16,   -32,  -64,   -8,   16,   32,   16,  -32,  -64,   -8,   16,   32],
		 		           [  16,    8,   -4, -128,  -64,   32,  112,   56,  -28,    8,     4,   -2,  -64,   -32,   16,   56,    28,  -14,   -4,   -2,    1,   32,   16,   -8,  -28,  -14,    7],
		 		           [   8,   64,    8,  -64, -512,  -64,   56,  448,   56,    4,    32,    4,  -32,  -256,  -32,   28,   224,   28,   -2,  -16,   -2,   16,  128,   16,  -14, -112,  -14],
		 		           [  -4,    8,   16,   32,  -64, -128,  -28,   56,  112,   -2,     4,    8,   16,   -32,  -64,  -14,    28,   56,    1,   -2,   -4,   -8,   16,   32,    7,  -14,  -28],
		 		           [  56,   28,  -14,  -64,  -32,   16,    8,    4,   -2,  448,   224, -112, -512,  -256,  128,   64,    32,  -16,   56,   28,  -14,  -64,  -32,   16,    8,    4,   -2],
		 		           [  28,  224,   28,  -32, -256,  -32,    4,   32,    4,  224,  1792,  224, -256, -2048, -256,   32,   256,   32,   28,  224,   28,  -32, -256,  -32,    4,   32,    4],
		 		           [ -14,   28,   56,   16,  -32,  -64,   -2,    4,    8, -112,   224,  448,  128,  -256, -512,  -16,    32,   64,  -14,   28,   56,   16,  -32,  -64,   -2,    4,    8],
		 		           [ -64,  -32,   16,  128,   64,  -32,  -64,  -32,   16, -512,  -256,  128, 1024,   512, -256, -512,  -256,  128,  -64,  -32,   16,  128,   64,  -32,  -64,  -32,   16],
		 		           [ -32, -256,  -32,   64,  512,   64,  -32, -256,  -32, -256, -2048, -256,  512,  4096,  512, -256, -2048, -256,  -32, -256,  -32,   64,  512,   64,  -32, -256,  -32],
		 		           [  16,  -32,  -64,  -32,   64,  128,   16,  -32,  -64,  128,  -256, -512, -256,   512, 1024,  128,  -256, -512,   16,  -32,  -64,  -32,   64,  128,   16,  -32,  -64],
		 		           [   8,    4,   -2,  -64,  -32,   16,   56,   28,  -14,   64,    32,  -16, -512,  -256,  128,  448,   224, -112,    8,    4,   -2,  -64,  -32,   16,   56,   28,  -14],
		 		           [   4,   32,    4,  -32, -256,  -32,   28,  224,   28,   32,   256,   32, -256, -2048, -256,  224,  1792,  224,    4,   32,    4,  -32, -256,  -32,   28,  224,   28],
		 		           [  -2,    4,    8,   16,  -32,  -64,  -14,   28,   56,  -16,    32,   64,  128,  -256, -512, -112,   224,  448,   -2,    4,    8,   16,  -32,  -64,  -14,   28,   56],
		 		           [ -28,  -14,    7,   32,   16,   -8,   -4,   -2,    1,   56,    28,  -14,  -64,   -32,   16,    8,     4,   -2,  112,   56,  -28, -128,  -64,   32,   16,    8,   -4],
		 		           [ -14, -112,  -14,   16,  128,   16,   -2,  -16,   -2,   28,   224,   28,  -32,  -256,  -32,    4,    32,    4,   56,  448,   56,  -64, -512,  -64,    8,   64,    8],
		 		           [   7,  -14,  -28,   -8,   16,   32,    1,   -2,   -4,  -14,    28,   56,   16,   -32,  -64,   -2,     4,    8,  -28,   56,  112,   32,  -64, -128,   -4,    8,   16],
		 		           [  32,   16,   -8,  -64,  -32,   16,   32,   16,   -8,  -64,   -32,   16,  128,    64,  -32,  -64,   -32,   16, -128,  -64,   32,  256,  128,  -64, -128,  -64,   32],
		 		           [  16,  128,   16,  -32, -256,  -32,   16,  128,   16,  -32,  -256,  -32,   64,   512,   64,  -32,  -256,  -32,  -64, -512,  -64,  128, 1024,  128,  -64, -512,  -64],
		 		           [  -8,   16,   32,   16,  -32,  -64,   -8,   16,   32,   16,   -32,  -64,  -32,    64,  128,   16,   -32,  -64,   32,  -64, -128,  -64,  128,  256,   32,  -64, -128],
		 		           [  -4,   -2,    1,   32,   16,   -8,  -28,  -14,    7,    8,     4,   -2,  -64,   -32,   16,   56,    28,  -14,   16,    8,   -4, -128,  -64,   32,  112,   56,  -28],
		 		           [  -2,  -16,   -2,   16,  128,   16,  -14, -112,  -14,    4,    32,    4,  -32,  -256,  -32,   28,   224,   28,    8,   64,    8,  -64, -512,  -64,   56,  448,   56],
		 		           [   1,   -2,   -4,   -8,   16,   32,    7,  -14,  -28,   -2,     4,    8,   16,   -32,  -64,  -14,    28,   56,   -4,    8,   16,   32,  -64, -128,  -28,   56,  112]]) / 1350.

		 Stt_R = np.array([ [ 112,   56,  -28,   56,    28,  -14,  -28,  -14,    7, -128,  -64,   32,  -64,   -32,   16,   32,   16,   -8,   16,    8,   -4,    8,     4,   -2,   -4,   -2,    1],
		 		           [  56,  448,   56,   28,   224,   28,  -14, -112,  -14,  -64, -512,  -64,  -32,  -256,  -32,   16,  128,   16,    8,   64,    8,    4,    32,    4,   -2,  -16,   -2],
		 		           [ -28,   56,  112,  -14,    28,   56,    7,  -14,  -28,   32,  -64, -128,   16,   -32,  -64,   -8,   16,   32,   -4,    8,   16,   -2,     4,    8,    1,   -2,   -4],
		 		           [  56,   28,  -14,  448,   224, -112,   56,   28,  -14,  -64,  -32,   16, -512,  -256,  128,  -64,  -32,   16,    8,    4,   -2,   64,    32,  -16,    8,    4,   -2],
		 		           [  28,  224,   28,  224,  1792,  224,   28,  224,   28,  -32, -256,  -32, -256, -2048, -256,  -32, -256,  -32,    4,   32,    4,   32,   256,   32,    4,   32,    4],
		 		           [ -14,   28,   56, -112,   224,  448,  -14,   28,   56,   16,  -32,  -64,  128,  -256, -512,   16,  -32,  -64,   -2,    4,    8,  -16,    32,   64,   -2,    4,    8],
		 		           [ -28,  -14,    7,   56,    28,  -14,  112,   56,  -28,   32,   16,   -8,  -64,   -32,   16, -128,  -64,   32,   -4,   -2,    1,    8,     4,   -2,   16,    8,   -4],
		 		           [ -14, -112,  -14,   28,   224,   28,   56,  448,   56,   16,  128,   16,  -32,  -256,  -32,  -64, -512,  -64,   -2,  -16,   -2,    4,    32,    4,    8,   64,    8],
		 		           [   7,  -14,  -28,  -14,    28,   56,  -28,   56,  112,   -8,   16,   32,   16,   -32,  -64,   32,  -64, -128,    1,   -2,   -4,   -2,     4,    8,   -4,    8,   16],
		 		           [-128,  -64,   32,  -64,   -32,   16,   32,   16,   -8,  256,  128,  -64,  128,    64,  -32,  -64,  -32,   16, -128,  -64,   32,  -64,   -32,   16,   32,   16,   -8],
		 		           [ -64, -512,  -64,  -32,  -256,  -32,   16,  128,   16,  128, 1024,  128,   64,   512,   64,  -32, -256,  -32,  -64, -512,  -64,  -32,  -256,  -32,   16,  128,   16],
		 		           [  32,  -64, -128,   16,   -32,  -64,   -8,   16,   32,  -64,  128,  256,  -32,    64,  128,   16,  -32,  -64,   32,  -64, -128,   16,   -32,  -64,   -8,   16,   32],
		 		           [ -64,  -32,   16, -512,  -256,  128,  -64,  -32,   16,  128,   64,  -32, 1024,   512, -256,  128,   64,  -32,  -64,  -32,   16, -512,  -256,  128,  -64,  -32,   16],
		 		           [ -32, -256,  -32, -256, -2048, -256,  -32, -256,  -32,   64,  512,   64,  512,  4096,  512,   64,  512,   64,  -32, -256,  -32, -256, -2048, -256,  -32, -256,  -32],
		 		           [  16,  -32,  -64,  128,  -256, -512,   16,  -32,  -64,  -32,   64,  128, -256,   512, 1024,  -32,   64,  128,   16,  -32,  -64,  128,  -256, -512,   16,  -32,  -64],
		 		           [  32,   16,   -8,  -64,   -32,   16, -128,  -64,   32,  -64,  -32,   16,  128,    64,  -32,  256,  128,  -64,   32,   16,   -8,  -64,   -32,   16, -128,  -64,   32],
		 		           [  16,  128,   16,  -32,  -256,  -32,  -64, -512,  -64,  -32, -256,  -32,   64,   512,   64,  128, 1024,  128,   16,  128,   16,  -32,  -256,  -32,  -64, -512,  -64],
		 		           [  -8,   16,   32,   16,   -32,  -64,   32,  -64, -128,   16,  -32,  -64,  -32,    64,  128,  -64,  128,  256,   -8,   16,   32,   16,   -32,  -64,   32,  -64, -128],
		 		           [  16,    8,   -4,    8,     4,   -2,   -4,   -2,    1, -128,  -64,   32,  -64,   -32,   16,   32,   16,   -8,  112,   56,  -28,   56,    28,  -14,  -28,  -14,    7],
		 		           [   8,   64,    8,    4,    32,    4,   -2,  -16,   -2,  -64, -512,  -64,  -32,  -256,  -32,   16,  128,   16,   56,  448,   56,   28,   224,   28,  -14, -112,  -14],
		 		           [  -4,    8,   16,   -2,     4,    8,    1,   -2,   -4,   32,  -64, -128,   16,   -32,  -64,   -8,   16,   32,  -28,   56,  112,  -14,    28,   56,    7,  -14,  -28],
		 		           [   8,    4,   -2,   64,    32,  -16,    8,    4,   -2,  -64,  -32,   16, -512,  -256,  128,  -64,  -32,   16,   56,   28,  -14,  448,   224, -112,   56,   28,  -14],
		 		           [   4,   32,    4,   32,   256,   32,    4,   32,    4,  -32, -256,  -32, -256, -2048, -256,  -32, -256,  -32,   28,  224,   28,  224,  1792,  224,   28,  224,   28],
		 		           [  -2,    4,    8,  -16,    32,   64,   -2,    4,    8,   16,  -32,  -64,  128,  -256, -512,   16,  -32,  -64,  -14,   28,   56, -112,   224,  448,  -14,   28,   56],
		 		           [  -4,   -2,    1,    8,     4,   -2,   16,    8,   -4,   32,   16,   -8,  -64,   -32,   16, -128,  -64,   32,  -28,  -14,    7,   56,    28,  -14,  112,   56,  -28],
		 		           [  -2,  -16,   -2,    4,    32,    4,    8,   64,    8,   16,  128,   16,  -32,  -256,  -32,  -64, -512,  -64,  -14, -112,  -14,   28,   224,   28,   56,  448,   56],
		 		           [   1,   -2,   -4,   -2,     4,    8,   -4,    8,   16,   -8,   16,   32,   16,   -32,  -64,   32,  -64, -128,    7,  -14,  -28,  -14,    28,   56,  -28,   56,  112]]) / 1350.

		 Dr_R = np.array([ [-3,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [-1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 1, -4,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0, -3,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  1, -4,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0, -3,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  1, -4,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  4, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  4, -1,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  3,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  4, -1,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  3,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  4, -1],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -4,  3]]) / 2.

		 Ds_R = np.array([ [-3,  0,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0, -3,  0,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0, -3,  0,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [-1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 1,  0,  0, -4,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  1,  0,  0, -4,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  1,  0,  0, -4,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  4,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -4,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -4,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -4,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  4,  0,  0, -1,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  4,  0,  0, -1,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  4,  0,  0, -1],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -4,  0,  0,  3,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -4,  0,  0,  3,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -4,  0,  0,  3]]) / 2.

		 Dt_R = np.array([ [-3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0, -1],
		 		          [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
		 		          [ 1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3,  0],
		 		          [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0,  0,  0,  0,  0,  0,  0,  0,  3]]) / 2.

	else:
		M_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R = (0, 0, 0, 0, 0, 0, 0)
	return (M_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R)