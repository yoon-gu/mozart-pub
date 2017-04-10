from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D, Dmatrices2D

def getMatrix(degree):
	"""
	Get FEM matrices on the reference triangle

	Paramters
		- ``degree`` (``int32``) : degree of polynomial

	Returns
		- ``M_R`` (``float64 array``) : Mass matrix on the reference triangle
		- ``Srr_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_r phi_i \partial_r phi_j dr)
		- ``Srs_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_r phi_i \partial_s phi_j dr)
		- ``Ssr_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_s phi_i \partial_r phi_j dr)
		- ``Sss_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_s phi_i \partial_s phi_j dr)
		- ``Dr_R`` (``float64 array``) : Differentiation matrix along r-direction
		- ``Ds_R`` (``float64 array``) : Differentiation matrix along s-direction

	Example
		>>> N = 1
		>>> M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R = getMatrix(N)
		>>> M_R
		array([[ 0.33333333,  0.16666667,  0.16666667],
		   [ 0.16666667,  0.33333333,  0.16666667],
		   [ 0.16666667,  0.16666667,  0.33333333]])
		>>> Srr_R
		array([[ 0.5, -0.5,  0. ],
		   [-0.5,  0.5,  0. ],
		   [ 0. ,  0. ,  0. ]])
		>>> Srs_R
		array([[  5.00000000e-01,  -9.80781986e-17,  -5.00000000e-01],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01],
		   [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00]])
		>>> Ssr_R
		array([[  5.00000000e-01,  -5.00000000e-01,   0.00000000e+00],
		   [ -9.80781986e-17,   9.80781986e-17,   0.00000000e+00],
		   [ -5.00000000e-01,   5.00000000e-01,   0.00000000e+00]])
		>>> Sss_R
		array([[  5.00000000e-01,  -9.80781986e-17,  -5.00000000e-01],
		   [ -9.80781986e-17,   1.92386661e-32,   9.80781986e-17],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01]])
		>>> Dr_R
		array([[-0.5,  0.5,  0. ],
		   [-0.5,  0.5,  0. ],
		   [-0.5,  0.5,  0. ]])
		>>> Ds_R
		array([[ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01],
		   [ -5.00000000e-01,   9.80781986e-17,   5.00000000e-01]])
	"""

	r, s = RefNodes_Tri(degree)
	V = Vandermonde2D(degree,r,s)
	invV = np.linalg.inv(V)
	M_R = np.dot(np.transpose(invV),invV)
	Dr_R, Ds_R = Dmatrices2D(degree, r, s, V)
	Srr_R = np.dot(np.dot(np.transpose(Dr_R),M_R),Dr_R)
	Srs_R = np.dot(np.dot(np.transpose(Dr_R),M_R),Ds_R)
	Ssr_R = np.dot(np.dot(np.transpose(Ds_R),M_R),Dr_R)
	Sss_R = np.dot(np.dot(np.transpose(Ds_R),M_R),Ds_R)
	return (M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R)

def compute_n4s(n4e):
	"""
	Get a matrix whose each row contains end points of corresponding side (or edge)

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``n4s`` (``int32 array``) : nodes for sides

	Example
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> n4s = compute_n4s(n4e)
		>>> n4s
		array([[1, 3],
		   [3, 0],
		   [1, 2],
		   [0, 1],
		   [2, 3]])
	"""
	allSides = np.vstack((np.vstack((n4e[:,[0,1]], n4e[:,[1,2]])),n4e[:,[2,0]]))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind = np.unique(x + y*1.0j, return_index=True)
	n4sInd = np.sort(ind)
	n4s = allSides[n4sInd,:]
	return n4s

def sample():
	from os import listdir
	from scipy.sparse import coo_matrix
	from scipy.sparse.linalg import spsolve

	folder = path.join(mz.__path__[0], 'samples', 'benchmark01')
	c4n_path = [file for file in listdir(folder) if 'c4n' in file][0]
	n4e_path = [file for file in listdir(folder) if 'n4e' in file][0]
	ind4e_path = [file for file in listdir(folder) if 'idx4e' in file][0]
	n4db_path = [file for file in listdir(folder) if 'n4sDb' in file][0]

	print(c4n_path)
	print(n4e_path)
	print(ind4e_path)
	print(n4db_path)

	c4n = np.fromfile(path.join(folder, c4n_path), dtype=np.float64)
	n4e = np.fromfile(path.join(folder, n4e_path), dtype=np.int32)
	ind4e = np.fromfile(path.join(folder, ind4e_path), dtype=np.int32)
	n4db = np.fromfile(path.join(folder, n4db_path), dtype=np.int32)

	print (c4n)
	print (n4e)
	print (ind4e)
	print (n4db)

	M_R = np.array([[2, 1, 1], [1, 2, 1],  [1, 1, 2]], dtype=np.float64) / 6.
	Srr_R = np.array([[1, -1, 0], [-1, 1, 0],  [0, 0, 0]], dtype=np.float64) / 2.
	Srs_R = np.array([[1, 0, -1], [-1, 0, 1],  [0, 0, 0]], dtype=np.float64) / 2.
	Ssr_R = np.array([[1, -1, 0], [0, 0, 0],  [-1, 1 ,0]], dtype=np.float64) / 2.
	Sss_R = np.array([[1, 0, -1], [0, 0, 0],  [-1, 0, 1]], dtype=np.float64) / 2.
	Dr_R = np.array([[-1, 1, 0], [-1, 1, 0],  [-1, 1, 0]], dtype=np.float64) / 2.
	Ds_R = np.array([[-1, 0, 1], [-1, 0 ,1],  [-1, 0 ,1]], dtype=np.float64) / 2.

	dim = 2
	nrNodes = int(len(c4n) / dim)
	nrElems = int(len(n4e) / 3)
	nrLocal = int(Srr_R.shape[0])

	f = np.ones((nrLocal * nrElems), dtype=np.float64) # RHS

	print((nrNodes, nrElems, dim, nrLocal))

	I = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	J = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.float64)
	b = np.zeros(nrNodes)

	Poison_2D = lib['Poisson_2D_Triangle'] # need the extern!!
	Poison_2D.argtypes = (c_void_p, c_void_p, c_void_p, c_int,
						c_void_p,
						c_void_p, c_void_p, c_void_p, c_void_p, c_int,
						c_void_p,
						c_void_p, c_void_p, c_void_p, c_void_p,)
	Poison_2D.restype = None
	Poison_2D(c_void_p(n4e.ctypes.data), c_void_p(ind4e.ctypes.data),
		c_void_p(c4n.ctypes.data), c_int(nrElems),
		c_void_p(M_R.ctypes.data),
		c_void_p(Srr_R.ctypes.data),
		c_void_p(Srs_R.ctypes.data),
		c_void_p(Ssr_R.ctypes.data),
		c_void_p(Sss_R.ctypes.data),
		c_int(nrLocal),
		c_void_p(f.ctypes.data),
		c_void_p(I.ctypes.data),
		c_void_p(J.ctypes.data),
		c_void_p(Alocal.ctypes.data),
		c_void_p(b.ctypes.data))


	STIMA_COO = coo_matrix((Alocal, (I, J)), shape=(nrNodes, nrNodes))
	STIMA_CSR = STIMA_COO.tocsr()

	dof = np.setdiff1d(range(0,nrNodes), n4db)

	# print STIMA_CSR

	x = np.zeros(nrNodes)
	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	print(x)

	# header_str = """
	# TITLE = "Example 2D Finite Element Triangulation Plot"
	# VARIABLES = "X", "Y", "U"
	# ZONE T="P_1", DATAPACKING=POINT, NODES={0}, ELEMENTS={1}, ZONETYPE=FETRIANGLE
	# """.format(nrNodes, nrElems)
	# print(header_str)

	# data_str = ""
	# for k in range(0, nrNodes):
	# 	data_str += "{0} {1} {2}\n".format(coord_x[k], coord_y[k], u[k])

	# np.savetxt(os.join(os.getcwd(), 'sample.dat'), (n4e+1).reshape((nrElems, 3)),
	# 	fmt='%d',
	# 	header=header_str + data_str, comments="")