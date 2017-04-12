import numpy as np
def compute_n4s(n4e):
	"""
	Get a matrix whose each row contains end points of the corresponding side (or edge)

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

def compute_s4e(n4e):
	"""
	Get a matrix whose each row contains three side numbers of the corresponding element

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``s4e`` (``int32 array``) : sides for elements

	Example
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> s4e = compute_s4e(n4e)
		>>> s4e
		array([[0, 1, 3],
		   [0, 2, 4]])
	"""
	allSides = np.vstack((np.vstack((n4e[:,[0,1]], n4e[:,[1,2]])),n4e[:,[2,0]]))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	s4e = sideNr[back].reshape(3,-1).transpose().astype('int')
	return s4e

def compute_e4s(n4e):
	"""
	Get a matrix whose each row contains two elements sharing the corresponding side
	If second column is -1, the corresponding side is on the boundary

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``e4s`` (``int32 array``) : elements for sides

	Example
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> e4s = compute_e4s(n4e)
		>>> e4s
		array([[ 0,  1],
		   [ 0, -1],
		   [ 1, -1],
		   [ 0, -1],
		   [ 1, -1]])
	"""
	allSides = np.vstack((np.vstack((n4e[:,[0,1]], n4e[:,[1,2]])),n4e[:,[2,0]]))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	n4sInd = np.sort(ind)

	nrElems = n4e.shape[0]
	elemNumbers = np.hstack((np.hstack((np.arange(0,nrElems),np.arange(0,nrElems))),np.arange(0,nrElems)))

	e4s=np.zeros((ind.size,2),int)
	e4s[:,0]=elemNumbers[n4sInd] + 1

	allElems4s=np.zeros(allSides.shape[0],int)
	tmp2 = np.bincount((back + 1),weights = (elemNumbers + 1))
	allElems4s[ind]=tmp2[1::]
	e4s[:,1] = allElems4s[n4sInd] - e4s[:,0]
	e4s=e4s-1
	return e4s

def refineUniformRed(c4n, n4e, n4Db, n4Nb):
	"""
	Refine a given mesh uniformly using the red refinement

	Paramters
		- ``c4n`` (``float64 array``) : coordinates for elements
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4Db`` (``int32 array``) : nodes for Difichlet boundary
		- ``n4Nb`` (``int32 array``) : nodes for Neumann boundary

	Returns
		- ``c4nNew`` (``float64 array``) : coordinates for element obtained from red refinement
		- ``n4eNew`` (``int32 array``) : nodes for element obtained from red refinement
		- ``n4DbNew`` (``int32 array``) : nodes for Dirichlet boundary obtained from red refinement
		- ``n4NbNew`` (``int32 array``) : nodes for Neumann boundary obtained from red refinement

	Example
		>>> c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
		>>> n4e = np.array([[1, 3, 0], [3, 1, 2]])
		>>> n4Db = np.array([[0, 1], [1, 2]])
		>>> n4Nb = np.array([[2, 3],[3, 0]])
		>>> c4nNew, n4eNew, n4DbNew, n4NbNew = refineUniformRed(c4n, n4e, n4Db, n4Nb)
		>>> c4nNew
		array([[ 0. ,  0. ],
		   [ 1. ,  0. ],
		   [ 1. ,  1. ],
		   [ 0. ,  1. ],
		   [ 0.5,  0.5],
		   [ 0. ,  0.5],
		   [ 1. ,  0.5],
		   [ 0.5,  0. ],
		   [ 0.5,  1. ]])
		>>> n4eNew
		array([[1, 4, 7],
		   [4, 3, 5],
		   [5, 7, 4],
		   [7, 5, 0],
		   [3, 4, 8],
		   [4, 1, 6],
		   [6, 8, 4],
		   [8, 6, 2]])
		>>> n4DbNew
		array([[0, 7],
		   [7, 1],
		   [1, 6],
		   [6, 2]])
		>>>n4NbNew
		array([[2, 8],
		   [8, 3],
		   [3, 5],
		   [5, 0]])
	"""
	nrNodes = c4n.shape[0]
	nrElems = n4e.shape[0]
	n4s = compute_n4s(n4e)
	nrSides = n4s.shape[0]
	from scipy.sparse import coo_matrix
	newNodes4s = coo_matrix((np.arange(0,nrSides)+nrNodes, (n4s[:,0], n4s[:,1])), shape=(nrNodes, nrNodes))
	newNodes4s = newNodes4s.tocsr()
	newNodes4s = newNodes4s + newNodes4s.transpose()

	mid4s = (c4n[n4s[:,0],:] + c4n[n4s[:,1],:]) * 0.5
	c4nNew = np.vstack((c4n, mid4s))

	n4eNew = np.zeros((4 * nrElems, 3), dtype=int)
	for elem in range(0,nrElems):
		nodes = n4e[elem,:]
		newNodes = np.array([newNodes4s[nodes[0],nodes[1]], newNodes4s[nodes[1],nodes[2]], newNodes4s[nodes[2],nodes[0]]])
		n4eNew[4*elem + np.arange(0,4),:] = np.array([[nodes[0], newNodes[0], newNodes[2]],
			[newNodes[0], nodes[1], newNodes[1]], [newNodes[1], newNodes[2], newNodes[0]],
			[newNodes[2], newNodes[1], nodes[2]]])

	n4DbNew = np.zeros((2 * n4Db.shape[0], 2), dtype = int)
	for side in range(0, n4Db.shape[0]):
		nodes = n4Db[side,:]
		newNodes = newNodes4s[nodes[0], nodes[1]]
		n4DbNew[2*side + np.arange(0,2),:] = np.array([[nodes[0], newNodes], [newNodes, nodes[1]]])

	n4NbNew = np.zeros((2 * n4Nb.shape[0], 2), dtype = int)
	for side in range(0, n4Nb.shape[0]):
		nodes = n4Nb[side,:]
		newNodes = newNodes4s[nodes[0], nodes[1]]
		n4NbNew[2*side + np.arange(0,2),:] = np.array([[nodes[0], newNodes], [newNodes, nodes[1]]])

	return (c4nNew, n4eNew, n4DbNew, n4NbNew)