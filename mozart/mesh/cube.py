import numpy as np

def cube(x1, x2, y1, y2, z1, z2, Mx, My, Mz, degree):
	"""
	Generates mesh information on the unit square [x1,x2]x[y1,y2].

	Parameters
		- ``x1`` (``float``) : coordinate of back point on the x-axis
		- ``x2`` (``float``) : coordinate of front point on the x-axis
		- ``y1`` (``float``) : coordinate of left point on the y-axis
		- ``y2`` (``float``) : coordinate of right point on the y-axis
		- ``z1`` (``float``) : coordinate of bottom point on the z-axis
		- ``z2`` (``float``) : coordinate of top point on the z-axis
		- ``Mx`` (``int``) : the number of elements along x-axis
		- ``My`` (``int``) : the number of elements along y-axis
		- ``Mz`` (``int``) : the number of elements along z-axis
		- ``degree`` (``int``) : polynomial degree for the approximate solution

	Returns
		- ``c4n`` (``float array``) : coordinates for nodes
		- ``ind4e`` (``int array``) : indices for elements
		- ``n4e`` (``int array``) : nodes for elements
		- ``n4Db`` (``int array``) : nodes for Dirichlet boundary

	Example
		>>> c4n, ind4e, n4e, n4Db = rectangle(0,1,0,1,0,1,2,2,2,1)
		>>> c4n
		array([[ 0.   0.   0. ], [ 0.5  0.   0. ], [ 1.   0.   0. ], [ 0.   0.5  0. ]
 			   [ 0.5  0.5  0. ], [ 1.   0.5  0. ], [ 0.   1.   0. ], [ 0.5  1.   0. ]
 			   [ 1.   1.   0. ], [ 0.   0.   0.5], [ 0.5  0.   0.5], [ 1.   0.   0.5]
 			   [ 0.   0.5  0.5], [ 0.5  0.5  0.5], [ 1.   0.5  0.5], [ 0.   1.   0.5]
 			   [ 0.5  1.   0.5], [ 1.   1.   0.5], [ 0.   0.   1. ], [ 0.5  0.   1. ]
			   [ 1.   0.   1. ], [ 0.   0.5  1. ], [ 0.5  0.5  1. ], [ 1.   0.5  1. ]
 			   [ 0.   1.   1. ], [ 0.5  1.   1. ], [ 1.   1.   1. ]])
	    >>> ind4e
	    array([[ 0  1  3  4  9 10 12 13]
			   [ 1  2  4  5 10 11 13 14]
			   [ 3  4  6  7 12 13 15 16]
			   [ 4  5  7  8 13 14 16 17]
			   [ 9 10 12 13 18 19 21 22]
			   [10 11 13 14 19 20 22 23]
			   [12 13 15 16 21 22 24 25]
			   [13 14 16 17 22 23 25 26]])
	    >>> n4e
	    array([[ 0  1  4  3  9 10 13 12]
			   [ 1  2  5  4 10 11 14 13]
			   [ 3  4  7  6 12 13 16 15]
			   [ 4  5  8  7 13 14 17 16]
			   [ 9 10 13 12 18 19 22 21]
			   [10 11 14 13 19 20 23 22]
			   [12 13 16 15 21 22 25 24]
			   [13 14 17 16 22 23 26 25]])
		>>> n4Db
		array([ 0  1  2  3  4  5  6  7  8  9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26])
	"""
	import numpy.matlib
	Nx = degree*Mx + 1
	Ny = degree*My + 1
	Nz = degree*Mz + 1

	c4n = np.zeros((Nx*Ny*Nz,3), dtype = np.float64)
	x = np.linspace(x1,x2,Nx)
	y = np.linspace(y1,y2,Ny)
	z = np.linspace(z1,z2,Nz)

	xx = np.matlib.repmat(np.matlib.repmat(x,1,y.size),1,z.size)
	yy = np.matlib.repmat(np.matlib.repmat(y,x.size,1),1,z.size)
	zz = np.matlib.repmat(z,x.size*y.size,1)

	c4n[:,0] = xx.flatten()
	c4n[:,1] = yy.transpose().flatten()
	c4n[:,2] = zz.transpose().flatten()

	ind4e = np.zeros((Mx*My*Mz,(degree+1)**3), dtype = np.int32)

	tmp1 = np.matlib.repmat(np.arange(1,degree*Mx+1,degree),My*Mz,1).transpose() + \
	np.matlib.repmat(np.arange(0,(degree*Mx+1)*((My-1)*degree+1),degree*(degree*Mx+1)),Mx,Mz)
	tmp2 = np.arange(0,(degree*Mx+1)*(My*degree+1)*((Mz-1)*degree+1),degree*(degree*Mx+1)*(degree*My+1))

	tmp1  = tmp1 + np.kron(tmp2,np.ones((Mx,My)))
	tmp1 = np.int32(tmp1)
	tmp1 = tmp1.transpose().flatten()

	tmp4 = np.kron(np.arange(0,degree*(degree*Mx+1)+1,degree*Mx+1),np.ones((Mx*My*Mz,degree+1))).transpose()
	tmp4 = np.matlib.repmat(tmp4,degree+1,1)

	tmp5 = np.arange(0,(degree*Mx+1)*(My*degree+1)*degree+1,(degree*Mx+1)*(degree*My+1))
	tmp5 = np.kron(tmp5,np.ones((Mx*My*Mz,(degree+1)**2)))

	tmp3 = np.matlib.repmat(tmp1,(degree+1)**3,1) + \
	np.matlib.repmat(np.arange(0,degree+1,1),Mx*My*Mz,(degree+1)**2).transpose() + tmp4 +\
	tmp5.transpose()

	ind4e = np.int32(tmp3.transpose() - 1)

	n4e = ind4e[[0,degree,(degree+1)**2-1,degree*(degree+1)],:]
	n4e = ind4e[:,[0, degree, (degree+1)**2-1, (degree+1)**2-degree-1,(degree+1)**2*degree,(degree+1)**2*degree+degree, (degree+1)**3-1, (degree+1)**3-degree-1]]

	x1Db = np.where(c4n[:, 0] == x1)[0]
	x2Db = np.where(c4n[:, 0] == x2)[0]
	y1Db = np.where(c4n[:, 1] == y1)[0]
	y2Db = np.where(c4n[:, 1] == y2)[0]
	z1Db = np.where(c4n[:, 2] == z1)[0]
	z2Db = np.where(c4n[:, 2] == z2)[0]
	n4Db = np.concatenate((x1Db,x2Db,y1Db,y2Db,z1Db,z2Db))
	n4Db = np.unique(n4Db)

	return (c4n, ind4e, n4e, n4Db)