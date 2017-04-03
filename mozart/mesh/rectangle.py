import numpy as np

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