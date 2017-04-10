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

def rectangle(x1, x2, y1, y2, Mx, My, degree):  
	"""  
	Generates mesh information on the unit square [x1,x2]x[y1,y2]. 

	Parameters  
		- ``x1`` (``float``) : coordinate of left point on the x-axis
		- ``x2`` (``float``) : coordinate of right point on the x-axis
		- ``y1`` (``float``) : coordinate of bottom point on the y-axis
		- ``y2`` (``float``) : coordinate of top point on the y-axis
		- ``Mx`` (``int``) : the number of elements along x-axis  
		- ``My`` (``int``) : the number of elements along y-axis  
		- ``degree`` (``int``) : polynomial degree for the approximate solution 

	Returns  
		- ``c4n`` (``float array``) : coordinates for nodes 
		- ``ind4e`` (``int array``) : indices for elements   
		- ``n4e`` (``int array``) : nodes for elements  		
		- ``n4Db`` (``int array``) : nodes for Dirichlet boundary 

	Example 
		>>> c4n, ind4e, n4e, n4Db = rectangle(0,1,0,1,2,2,1)
		>>> c4n 
		array([[ 0.   0. ]
 			   [ 0.5  0. ]
 			   [ 1.   0. ]
 			   [ 0.   0.5]
 			   [ 0.5  0.5]
 			   [ 1.   0.5]
 			   [ 0.   1. ]
 			   [ 0.5  1. ]
			   [ 1.   1. ]])
	    >>> ind4e
	    array([[0 1 3 4]
 			   [1 2 4 5]
 			   [3 4 6 7]
 			   [4 5 7 8]])
	    >>> n4e
	    array([[0 1 4 3]
 			   [1 2 5 4]
 			   [3 4 7 6]
 			   [4 5 8 7]])
		>>> n4Db 
		array([0 1 2 3 5 6 7 8])
	""" 
	import numpy as np
	import numpy.matlib 

	Nx = degree*Mx + 1
	Ny = degree*My + 1

	c4n = np.zeros((2,Nx*Ny), dtype = np.float64)
	x = np.linspace(x1,x2,Nx)
	y = np.linspace(y1,y2,Ny)
	xx = np.matlib.repmat(x,1,Ny)
	yy = np.kron(y,np.ones((1,Nx), dtype = np.float64))
	c4n[0] = xx
	c4n[1] = yy

	ind4e = np.zeros(((degree+1)**2,Mx*My), dtype = np.int32)
	tmp1 = np.matlib.repmat(np.arange(0,Mx*degree,degree),1,My) 
	tmp2 =np.matlib.repmat(np.arange(0,(Mx*degree+1)*((My-1)*degree+1)-1,degree*(Mx*degree+1)),Mx,1)  
	tmp1 = tmp1 + np.transpose(tmp2).flatten() 
	for j in range(0,degree+1): 
		ind4e[j*(degree+1)+np.arange(0,degree+1,1),:] = np.matlib.repmat(tmp1+j*(degree*Mx+1),degree+1,1) + np.transpose(np.matlib.repmat(np.arange(0,degree+1,1),Mx*My,1)) 

	n4e = ind4e[[0,degree,(degree+1)**2-1,degree*(degree+1)],:] 


	btm = np.arange(0,Nx,1)
	top = np.arange(Nx*(Ny-1),Nx*Ny,1)
	left = np.arange(0,Nx*(Ny-1)+1,Nx)
	right = np.arange(Nx-1,Nx*Ny,Nx) 
	n4Db = np.unique(np.concatenate((btm,top,left,right)))

	return (c4n.transpose(), ind4e.transpose(), n4e.transpose(), n4Db)