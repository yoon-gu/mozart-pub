import numpy as np
def unit_interval(N):
	"""
	Computes the coordinates of nodes and elements.
	
	Parameters
		- ``N`` (``int``) : Number of nodes

	Returns
		- ``c4n`` (``float array``) : coordinates of nodes
		- ``n4e`` (``int array``) : elements

	Example
		>>> c4n, n4e = unit_interval(4)
		>>> c4n 
		array([ 0.        ,  0.33333333,  0.66666667,  1.        ])
		>>> n4e
		array([[0, 1],
		   [1, 2],
		   [2, 3]])
	"""
	c4n = np.linspace(0, 1, N)
	n4e = np.array([[item,item+1] for item in range(0,N-1)], dtype=np.int32)
	return (c4n, n4e)

def interval(a, b, M, degree):
	"""
	Generates mesh information on an interval [a,b].
	
	Parameters
		- ``a`` (``float``) : coordinate of left-end point of the interval
		- ``b`` (``float``) : coordinate of right-end point of the interval
		- ``M`` (``int``) : the number of elements
		- ``degree`` (``int``) : polynomial degree for the approximate solution

	Returns
		- ``c4n`` (``float array``) : coordinates for nodes
		- ``n4e`` (``int array``) : nodes for elements
		- ``n4db`` (``int array``) : nodes for Dirichlet boundary
		- ``ind4e`` (``int array``) : indices for elements

	Example
		>>> c4n, n4e, n4db, ind4e = interval(0,1,4,2)
		>>> c4n 
		array([ 0.   ,  0.125,  0.25 ,  0.375,  0.5  ,  0.625,  0.75 ,  0.875,  1.   ])
		>>> n4e
		array([[0, 2],
		   [2, 4],
		   [4, 6],
		   [6, 8]])
		>>> n4db
		array([0, 8])
		>>> ind4e
		array([[0, 1, 2],
		   [2, 3, 4],
		   [4, 5, 6],
		   [6, 7, 8]])
	"""
	nrNodes = M*degree + 1; # the number of nodes on the mesh in terms of M and degree
	c4n = np.linspace(a, b, nrNodes)
	n4e = np.array([[degree*item, degree*item+degree] for item in range(0,M)], dtype=np.int32)
	n4db = np.array([0, nrNodes - 1])
	ind4e = np.array([np.arange(degree*item,degree*item+degree+1,1) for item in range(0,M)], dtype=np.int32)
	return (c4n, n4e, n4db, ind4e)

def unit_square(M,degree):
	"""
	Generates mesh information on the unit square [0,1]x[0,1].
	
	Parameters
		- ``M`` (``int``) : the number of elements along axis
		- ``degree`` (``int``) : polynomial degree for the approximate solution

	Returns
		- ``c4n`` (``float array``) : coordinates for nodes
		- ``n4e`` (``int array``) : nodes for elements
		- ``ind4e`` (``int array``) : indices for elements
		- ``n4Db`` (``int array``) : nodes for Dirichlet boundary

	Example
	"""
	import numpy.matlib

	nrNodes = (M*degree + 1)**2
	nrLocal = (degree + 1)**2
	nrElems = M**2

	c4n = np.zeros((2,(M*degree + 1)**2), dtype = np.float64)
	tmp = np.linspace(0,1,M*degree + 1)
	c4n[0] = np.matlib.repmat(tmp,1,3)
	c4n[1] = np.transpose(np.matlib.repmat(tmp,3,1)).flatten()

	ind4e = np.zeros((nrLocal,nrElems), dtype = np.int32)
	tmp = np.matlib.repmat(np.arange(0,M*degree,degree),1,M)
	tmp1 =np.matlib.repmat(np.arange(0,(M*degree+1)*((M-1)*degree+1)-1,degree*(M*degree+1)),M,1) 
	tmp = tmp + np.transpose(tmp1).flatten()
	for j in range(0,degree+1):
		ind4e[j*(degree+1)+np.arange(0,degree+1,1),:] = np.matlib.repmat(tmp+j*(degree*M+1),degree+1,1) + np.transpose(np.matlib.repmat(np.arange(0,degree+1,1),M**2,1))

	n4e = ind4e[[0,degree,(degree+1)**2-1,degree*(degree+1)],:]

	btm = np.arange(0,degree*M+1,1)
	top = np.arange((degree*M+1)*M*degree,(degree*M+1)**2,1)
	left = np.arange(0,(degree*M+1)*M*degree+1,degree*M+1)
	right = np.arange(degree*M,(degree*M+1)**2,degree*M+1)
	n4Db = np.unique((btm,top,left,right))

	return (c4n, n4e, ind4e, n4Db)