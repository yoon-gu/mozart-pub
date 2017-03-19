def getMatrix1D(degree):
	"""
	Get FEM matrices on the reference domain
	"""
	M_R = None
	S_R = None
	D_R = None
	if degree is 1:
		M_R = [2 1; 1 2] / 3.
		S_R = [1 -1; -1 1] / 2.
		D_R = [-1 1; -1 1] / 2.
	if degree is 2:
		M_R = [4 2 -1; 2 16 2; -1 2 4] / 15.
		S_R = [7 -8 1; -8 16 -8; 1 -8 7] / 6.
		D_R = [-3 4 -1; -1 0 1; 1 -4 3] / 2.

	return (M_R, S_R, D_R)

def one_dim(c4n, n4e, n4sDb, f):
	print("one_dim is called.")

def two_dim(c4n, n4e, n4sDb, f):
	print("two_dim is called.")

def three_dim(c4n, n4e, n4sDb, f):
	print("trhee_dim is called.")

# def sample():
# 	from os import listdir, path
# 	from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
# 	import numpy as np
# 	from scipy.sparse import coo_matrix
# 	from scipy.sparse.linalg import spsolve

# 	dllpath = path.join(mz.__path__[0], 'libmozart.so')

# 	lib = CDLL(dllpath)
# 	folder = path.join(mz.__path__[0], 'samples', 'benchmark01')
# 	c4n_path = [file for file in listdir(folder) if 'c4n' in file][0]
# 	n4e_path = [file for file in listdir(folder) if 'n4e' in file][0]
# 	ind4e_path = [file for file in listdir(folder) if 'idx4e' in file][0]
# 	n4db_path = [file for file in listdir(folder) if 'n4sDb' in file][0]

# 	print(c4n_path)
# 	print(n4e_path)
# 	print(ind4e_path)
# 	print(n4db_path)

# 	c4n = np.fromfile(path.join(folder, c4n_path), dtype=np.float64)
# 	n4e = np.fromfile(path.join(folder, n4e_path), dtype=np.int32)
# 	ind4e = np.fromfile(path.join(folder, ind4e_path), dtype=np.int32)
# 	n4db = np.fromfile(path.join(folder, n4db_path), dtype=np.int32)

# 	print (c4n)
# 	print (n4e)
# 	print (ind4e)
# 	print (n4db)

# 	M_R = np.array([[2, 1, 1], [1, 2, 1],  [1, 1, 2]], dtype=np.float64) / 6.
# 	Srr_R = np.array([[1, -1, 0], [-1, 1, 0],  [0, 0, 0]], dtype=np.float64) / 2.
# 	Srs_R = np.array([[1, 0, -1], [-1, 0, 1],  [0, 0, 0]], dtype=np.float64) / 2.
# 	Ssr_R = np.array([[1, -1, 0], [0, 0, 0],  [-1, 1 ,0]], dtype=np.float64) / 2.
# 	Sss_R = np.array([[1, 0, -1], [0, 0, 0],  [-1, 0, 1]], dtype=np.float64) / 2.
# 	Dr_R = np.array([[-1, 1, 0], [-1, 1, 0],  [-1, 1, 0]], dtype=np.float64) / 2.
# 	Ds_R = np.array([[-1, 0, 1], [-1, 0 ,1],  [-1, 0 ,1]], dtype=np.float64) / 2.

# 	dim = 2
# 	nrNodes = int(len(c4n) / dim)
# 	nrElems = int(len(n4e) / 3)
# 	nrLocal = int(Srr_R.shape[0])

# 	f = np.ones((nrLocal * nrElems), dtype=np.float64) # RHS

# 	print((nrNodes, nrElems, dim, nrLocal))

# 	I = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
# 	J = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
# 	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.float64)
# 	b = np.zeros(nrNodes)

# 	Poison_2D = lib['Poisson_2D_Triangle'] # need the extern!!
# 	Poison_2D.argtypes = (c_void_p, c_void_p, c_void_p, c_int,
# 						c_void_p,
# 						c_void_p, c_void_p, c_void_p, c_void_p, c_int,
# 						c_void_p,
# 						c_void_p, c_void_p, c_void_p, c_void_p,)
# 	Poison_2D.restype = None
# 	Poison_2D(c_void_p(n4e.ctypes.data), c_void_p(ind4e.ctypes.data),
# 		c_void_p(c4n.ctypes.data), c_int(nrElems),
# 		c_void_p(M_R.ctypes.data),
# 		c_void_p(Srr_R.ctypes.data),
# 		c_void_p(Srs_R.ctypes.data),
# 		c_void_p(Ssr_R.ctypes.data),
# 		c_void_p(Sss_R.ctypes.data),
# 		c_int(nrLocal),
# 		c_void_p(f.ctypes.data),
# 		c_void_p(I.ctypes.data),
# 		c_void_p(J.ctypes.data),
# 		c_void_p(Alocal.ctypes.data),
# 		c_void_p(b.ctypes.data))


# 	STIMA_COO = coo_matrix((Alocal, (I, J)), shape=(nrNodes, nrNodes))
# 	STIMA_CSR = STIMA_COO.tocsr()

# 	dof = np.setdiff1d(range(0,nrNodes), n4db)

# 	# print STIMA_CSR

# 	x = np.zeros(nrNodes)
# 	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
# 	print(x)

# 	import matplotlib.pyplot as plt
# 	from mpl_toolkits.mplot3d import Axes3D

# 	fig = plt.figure()
# 	ax = fig.gca(projection='3d')

# 	coord_x = c4n[0::2]
# 	coord_y = c4n[1::2]
# 	u = x
# 	ax.plot_trisurf(coord_x, coord_y, u, triangles=n4e.reshape((nrElems, 3)), cmap=plt.cm.Spectral)
# 	plt.show()

# 	# header_str = """
# 	# TITLE = "Example 2D Finite Element Triangulation Plot"
# 	# VARIABLES = "X", "Y", "U"
# 	# ZONE T="P_1", DATAPACKING=POINT, NODES={0}, ELEMENTS={1}, ZONETYPE=FETRIANGLE
# 	# """.format(nrNodes, nrElems)
# 	# print(header_str)

# 	# data_str = ""
# 	# for k in range(0, nrNodes):
# 	# 	data_str += "{0} {1} {2}\n".format(coord_x[k], coord_y[k], u[k])

# 	# np.savetxt(os.join(os.getcwd(), 'sample.dat'), (n4e+1).reshape((nrElems, 3)),
# 	# 	fmt='%d',
# 	# 	header=header_str + data_str, comments="")