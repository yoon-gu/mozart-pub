from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np

# OS Detection Code
prefix = "linux"
if platform == "linux" or platform == "linux32":
	prefix = "linux"
elif platform == "darwin":
	prefix = "osx"
elif platform == "win32":
	prefix = "win64"

dllpath = path.join(mz.__path__[0], prefix + '_' + 'libmozart.so')
lib = CDLL(dllpath)

def getMatrix1D(degree):
	"""
	Get FEM matrices on the reference domain

	Paramters
		- ``degree`` : degree of polynomial

	Returns
		- ``M_R`` : 
		- ``S_R`` : 
		- ``D_R`` :  

	"""
	M_R = None
	S_R = None
	D_R = None
	if degree is 1:
		M_R = np.array([[ 2,  1], [ 1, 2]]) / 3.
		S_R = np.array([[ 1, -1], [-1, 1]]) / 2.
		D_R = np.array([[-1,  1], [-1, 1]]) / 2.
	if degree is 2:
		M_R = np.array([[ 4,  2, -1], [ 2, 16,  2], [-1,  2, 4]]) / 15.
		S_R = np.array([[ 7, -8,  1], [-8, 16, -8], [ 1, -8, 7]]) / 6.
		D_R = np.array([[-3,  4, -1], [-1,  0,  1], [ 1, -4, 3]]) / 2.

	return (M_R, S_R, D_R)

def nJacobiP(x=np.array([0]), alpha=0, beta=0, degree=0):
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
		>>> from mozart.poisson.solve import nJacobiP
		>>> P = nJacobiP(x,0,0,N)
		>>> print(P)
		array([ 1.58113883, -0.79056942,  1.58113883])	
	"""
	Pn = np.zeros((degree+1,x.size),float)
	Pn[0,:] = np.sqrt(2**(-alpha-beta-1) * np.math.gamma(alpha+beta+2) / ((np.math.gamma(alpha + 1) * np.math.gamma(beta + 1))))

	if degree == 0:
		P = Pn
	else:
		Pn[1,:] = np.multiply(Pn[0,:]*np.sqrt((alpha+beta+3.0)/((alpha+1)*(beta+1))),((alpha+beta+2)*x+(alpha-beta)))/2
		a_n = 2.0/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
		for n in range(2,degree+1):
			anew=2.0/(2*n+alpha+beta)*np.sqrt(n*(n+alpha+beta)*(n+alpha)*(n+beta)/((2*n+alpha+beta-1)*(2*n+alpha+beta+1)))
			b_n=-(alpha**2-beta**2)/((2*(n-1)+alpha+beta)*(2*(n-1)+alpha+beta+2.0))
			Pn[n,:]=(np.multiply((x-b_n),Pn[n-1,:])-a_n*Pn[n-2,:])/anew
			a_n=anew

	P = Pn[degree,:]
	return P

def DnJacobiP(x=np.array([0]), alpha=0, beta=0, degree=0):
	"""
	Evaluate the derivative of the normalized Jacobi polynomial of type alpha, beta > -1 at point x for order n

	Paramters
		- ``x`` (``float64 array``) : variable x
		- ``alpha`` (``int32``) : superscript alpha of normalized Jacobi polynomial
		- ``beta`` (``int32``) : superscript beta of normalized Jacobi polynomial
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``dP`` (``float64 array``) : the value of the derivative of the normalized Jacobi polynomial at x
									   according to alpha, beta, degree
	
	Example
		>>> N = 2
		>>> x = np.array([-1, 0, 1])
		>>> from mozart.poisson.solve import DnJacobiP
		>>> dP = DnJacobiP(x,0,0,N)
		>>> print(dP)
		array([-4.74341649,  0.        ,  4.74341649])	
	"""
	length = x.size
	dP = np.zeros(length,float)
	if degree == 0:
		dP[:] = 0
	else:
		dP[:] = np.sqrt(degree*(degree+alpha+beta+1))*nJacobiP(x,alpha+1,beta+1,degree-1)
	return dP

def one_dim(c4n, n4e, n4Db, f, u_D, degree = 1):
	"""
	Computes the coordinates of nodes and elements.
	
	Parameters
		- ``c4n`` (``float64 array``) : coordinates
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4Db`` (``int32 array``) : Dirichlet boundary nodes
		- ``f`` (``lambda``) : source term 
		- ``u_D`` (``lambda``) : Dirichlet boundary condition
		- ``degree`` (``int32``) : Polynomial degree

	Returns
		- ``x`` (``float64 array``) : solution

	Example
		>>> N = 3
		>>> c4n, n4e = unit_interval(N)
		>>> n4Db = [0, N-1]
		>>> f = lambda x: np.ones_like(x)
		>>> u_D = lambda x: np.zeros_like(x)
		>>> from mozart.poisson.solve import one_dim
		>>> x = one_dim(c4n, n4e, n4Db, f, u_D)
		>>> print(x)
		array([ 0.   ,  0.125,  0.   ])
	"""
	from mozart.poisson.solve import getMatrix1D
	M_R, S_R, D_R = getMatrix1D(degree)
	fval = f(c4n[n4e].flatten())
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
	Poison_1D(c_void_p(n4e.ctypes.data), c_void_p(n4e.ctypes.data),
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

	dof = np.setdiff1d(range(0,nrNodes), n4Db)

	x = np.zeros(nrNodes)
	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	return x

def two_dim(c4n, n4e, n4sDb, f):
	print("two_dim is called.")

def three_dim(c4n, n4e, n4sDb, f):
	print("trhee_dim is called.")

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