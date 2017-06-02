from sys import platform
from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz
import numpy as np
from mozart.common.etc import prefix_by_os
dllpath = path.join(mz.__path__[0], prefix_by_os(platform) + '_' + 'libmozart.so')
lib = CDLL(dllpath)

from mozart.poisson.fem.common import nJacobiP, DnJacobiP, VandermondeM1D, Dmatrix1D, RefNodes_Rec, RefNodes_Cube

def compute_n4s(n4e):
	"""
	Get a matrix whose each row contains end points of the corresponding side (or edge)

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``n4s`` (``int32 array``) : nodes for sides

	Example
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> n4s = compute_n4s(n4e)
		>>> n4s
		array([[ 0,  1],
		   [ 1,  2],
		   [ 1,  4],
		   [ 2,  5],
		   [ 4,  3],
		   [ 5,  4],
		   [ 3,  0],
		   [ 0,  6],
		   [ 1,  7],
		   [ 2,  8],
		   [ 4, 10],
		   [ 5, 11],
		   [ 3,  9],
		   [ 6,  7],
		   [ 7,  8],
		   [ 7, 10],
		   [ 8, 11],
		   [10,  9],
		   [11, 10],
		   [ 9,  6]])
	"""
	allSides = np.array([n4e[:,[0,1]], n4e[:,[1,2]], n4e[:,[2,3]], n4e[:,[3,0]],
		n4e[:,[0,4]], n4e[:,[1,5]], n4e[:,[2,6]], n4e[:,[3,7]],
		n4e[:,[4,5]], n4e[:,[5,6]], n4e[:,[6,7]], n4e[:,[7,4]]])
	allSides = np.reshape(allSides,(12*n4e.shape[0],-1))
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
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> s4e = compute_s4e(n4e)
		>>> s4e
		array([[ 0,  2,  4,  6,  7,  8, 10, 12, 13, 15, 17, 19],
		   [ 1,  3,  5,  2,  8,  9, 11, 10, 14, 16, 18, 15]])
	"""
	allSides = np.array([n4e[:,[0,1]], n4e[:,[1,2]], n4e[:,[2,3]], n4e[:,[3,0]],
		n4e[:,[0,4]], n4e[:,[1,5]], n4e[:,[2,6]], n4e[:,[3,7]],
		n4e[:,[4,5]], n4e[:,[5,6]], n4e[:,[6,7]], n4e[:,[7,4]]])
	allSides = np.reshape(allSides,(12*n4e.shape[0],-1))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	s4e = sideNr[back].reshape(12,-1).transpose().astype('int')
	return s4e

def compute_n4f(n4e):
	"""
	Get a matrix whose each row contains end points of the corresponding face

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``n4f`` (``int32 array``) : nodes for faces

	Example
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> n4f = compute_n4f(n4e)
		>>> n4f
		array([[ 0,  1,  4,  3],
		   [ 1,  2,  5,  4],
		   [ 0,  1,  7,  6],
		   [ 1,  2,  8,  7],
		   [ 1,  4, 10,  7],
		   [ 2,  5, 11,  8],
		   [ 4,  3,  9, 10],
		   [ 5,  4, 10, 11],
		   [ 3,  0,  6,  9],
		   [ 6,  7, 10,  9],
		   [ 7,  8, 11, 10]])
	"""
	allFaces = np.array([n4e[:,[0,1,2,3]], n4e[:,[0,1,5,4]], n4e[:,[1,2,6,5]],
		n4e[:,[2,3,7,6]], n4e[:,[3,0,4,7]], n4e[:,[4,5,6,7]]])
	allFaces = np.reshape(allFaces,(6*n4e.shape[0],-1))
	tmp=np.sort(allFaces)
	tmp=np.ascontiguousarray(tmp)
	_, ind = np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True)
	n4fInd = np.sort(ind)
	n4f = allFaces[n4fInd,:]
	return n4f

def compute_f4e(n4e):
	"""
	Get a matrix whose each row contains six face numbers of the corresponding element

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``f4e`` (``int32 array``) : faces for elements

	Example
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> f4e = compute_f4e(n4e)
		>>> f4e
		array([[ 0,  2,  4,  6,  8,  9],
		   [ 1,  3,  5,  7,  4, 10]])
	"""
	allFaces = np.array([n4e[:,[0,1,2,3]], n4e[:,[0,1,5,4]], n4e[:,[1,2,6,5]],
		n4e[:,[2,3,7,6]], n4e[:,[3,0,4,7]], n4e[:,[4,5,6,7]]])
	allFaces = np.reshape(allFaces,(6*n4e.shape[0],-1))
	tmp=np.sort(allFaces)
	tmp=np.ascontiguousarray(tmp)
	_, ind, back = np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True, return_inverse=True)
	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	f4e = sideNr[back].reshape(6,-1).transpose().astype('int')
	return f4e

def compute_e4f(n4e):
	"""
	Get a matrix whose each row contains two elements sharing the corresponding face
	If second column is -1, the corresponding face is on the boundary

	Paramters
		- ``n4e`` (``int32 array``) : nodes for elements

	Returns
		- ``e4f`` (``int32 array``) : elements for faces

	Example
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> e4f = compute_e4f(n4e)
		>>> e4f
		array([[ 0, -1],
		   [ 0, -1],
		   [ 0, -1],
		   [ 0, -1],
		   [ 0,  1],
		   [ 0, -1],
		   [ 1, -1],
		   [ 1, -1],
		   [ 1, -1],
		   [ 1, -1],
		   [ 1, -1]])
	"""
	allFaces = np.array([n4e[:,[0,1,2,3]], n4e[:,[0,1,5,4]], n4e[:,[1,2,6,5]],
		n4e[:,[2,3,7,6]], n4e[:,[3,0,4,7]], n4e[:,[4,5,6,7]]])
	allFaces = np.reshape(allFaces,(6*n4e.shape[0],-1))
	tmp=np.sort(allFaces)
	tmp=np.ascontiguousarray(tmp)
	_, ind, back = np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True, return_inverse=True)
	n4fInd = np.sort(ind)

	nrElems = n4e.shape[0]
	import numpy.matlib
	elemNumbers = np.matlib.repmat(np.arange(0,nrElems),6,1).flatten('F')

	e4f=np.zeros((ind.size,2),int)
	e4f[:,0]=elemNumbers[n4fInd] + 1

	allElems4s=np.zeros(allFaces.shape[0],int)
	tmp2 = np.bincount((back + 1),weights = (elemNumbers + 1))
	allElems4s[ind]=tmp2[1::]
	e4f[:,1] = allElems4s[n4fInd] - e4f[:,0]
	e4f=e4f-1
	return e4f

def getIndex(degree, c4n, n4e, n4fDb, n4fNb):
	"""
	Get indices on each element

	Paramters
		- ``degree`` (``int32``) : degree of polynomial
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``n4fDb`` (``int32 array``) : nodes for faces on Dirichlet boundary
		- ``n4fNb`` (``int32 array``) : nodes for faces on Neumann boundary

	Returns
		- ``c4nNew`` (``float64 array``) : coordinates for all nodes (vetex, side and interior nodes)
		- ``ind4e`` (``int32 array``) : indices on each element
		- ``ind4Db`` (``int32 array``) : indices on Dirichlet boundary
		- ``ind4Nb`` (``int32 array``) : indices on Neumann boundary

	Example
		>>> N = 3
		>>> c4n = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 1., 0.],
							[1., 1., 0.], [2., 1., 0.], [0., 0., 1.], [1., 0., 1.],
							[2., 0., 1.], [0., 1., 1.], [1., 1., 1.], [2., 1., 1.]])
		>>> n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
		>>> n4fDb = np.array([[0, 1, 7, 6], [1, 2, 8, 7], [2, 5, 11, 8], [5, 4, 10, 11],
							  [4, 3, 9, 10], [3, 0, 6, 9], [6, 7, 10, 9], [7, 8, 11, 10]])
		>>> n4fNb = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
		>>> c4nNew, ind4e, ind4Db, ind4Nb = getIndex(N, c4n, n4e, n4fDb, n4fNb)
	"""
	allSides = np.array([n4e[:,[0,1]], n4e[:,[1,2]], n4e[:,[2,3]], n4e[:,[3,0]],
		n4e[:,[0,4]], n4e[:,[1,5]], n4e[:,[2,6]], n4e[:,[3,7]],
		n4e[:,[4,5]], n4e[:,[5,6]], n4e[:,[6,7]], n4e[:,[7,4]]])
	allSides = np.reshape(allSides,(12*n4e.shape[0],-1))
	tmp=np.sort(allSides)
	x, y = tmp.T
	_, ind, back = np.unique(x + y*1.0j, return_index=True, return_inverse=True)
	n4sInd = np.sort(ind)
	n4s = allSides[n4sInd,:]

	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	s4e = sideNr[back].reshape(12,-1).transpose().astype('int')


	allFaces = np.array([n4e[:,[0,1,2,3]], n4e[:,[0,1,5,4]], n4e[:,[1,2,6,5]],
		n4e[:,[2,3,7,6]], n4e[:,[3,0,4,7]], n4e[:,[4,5,6,7]]])
	allFaces = np.reshape(allFaces,(6*n4e.shape[0],-1))
	tmp=np.sort(allFaces)
	tmp=np.ascontiguousarray(tmp)
	_, ind, back = np.unique(tmp.view([('',tmp.dtype)]*tmp.shape[1]),return_index=True, return_inverse=True)

	n4fInd = np.sort(ind)
	n4f = allFaces[n4fInd,:]

	sortInd = ind.argsort()
	sideNr = np.zeros(ind.size, dtype = int)
	sideNr[sortInd] = np.arange(0,ind.size)
	f4e = sideNr[back].reshape(6,-1).transpose().astype('int')

	nrElems = n4e.shape[0]
	import numpy.matlib
	elemNumbers = np.matlib.repmat(np.arange(0,nrElems),6,1).flatten()

	e4f=np.zeros((ind.size,2),int)
	e4f[:,0]=elemNumbers[n4fInd] + 1

	allElems4s=np.zeros(allFaces.shape[0],int)
	tmp2 = np.bincount((back + 1),weights = (elemNumbers + 1))
	allElems4s[ind]=tmp2[1::]
	e4f[:,1] = allElems4s[n4fInd] - e4f[:,0]
	e4f=e4f-1

	S_ind = np.zeros((nrElems,12), dtype = int)
	for j in range(0,nrElems):
		if (n4s[s4e[j,0],:] == n4e[j,[0,1]]).all():
			S_ind[j,0] = 1

		if (n4s[s4e[j,1],:] == n4e[j,[1,2]]).all():
			S_ind[j,1] = 1

		if (n4s[s4e[j,2],:] == n4e[j,[2,3]]).all():
			S_ind[j,2] = 1

		if (n4s[s4e[j,3],:] == n4e[j,[3,0]]).all():
			S_ind[j,3] = 1

		if (n4s[s4e[j,8],:] == n4e[j,[4,5]]).all():
			S_ind[j,8] = 1

		if (n4s[s4e[j,9],:] == n4e[j,[5,6]]).all():
			S_ind[j,9] = 1

		if (n4s[s4e[j,10],:] == n4e[j,[6,7]]).all():
			S_ind[j,10] = 1

		if (n4s[s4e[j,11],:] == n4e[j,[7,4]]).all():
			S_ind[j,11] = 1

	S_ind[:,4:8] = 1
	S_ind = (np.outer(np.arange(0,degree-1),S_ind) + np.outer(np.arange(degree-2,-1,-1),1-S_ind)).flatten('F')
	S_ind = np.reshape(S_ind,(nrElems,(degree-1)*12))

	nru = (degree+1)**3
	nri = (degree-1)**3
	nrf = (degree-1)**2
	nrs = degree-1
	nrNodes = c4n.shape[0]
	nNS = nrNodes + nrs * n4s.shape[0]
	nNSF = nNS + nrf * n4f.shape[0]
	AllNodes = nNSF + nri * nrElems

	BDindex_S = np.array([np.arange(1,degree), np.arange(2*degree+1,degree*degree+degree,degree+1), np.arange((degree+1)**2-2,degree*(degree+1),-1),
						  np.arange((degree-1)*(degree+1),degree,-(degree+1)), np.arange((degree+1)**2,(degree+1)**2*(degree-1)+1,(degree+1)**2),
						  np.arange((degree+1)**2+degree,(degree+1)**2*(degree-1)+degree+1,(degree+1)**2),
						  np.arange(2*(degree+1)**2-1,degree*(degree+1)**2,(degree+1)**2),
						  np.arange((degree+1)**2+degree*(degree+1),(degree-1)*(degree+1)**2+degree*(degree+1)+1,(degree+1)**2),
						  degree*(degree+1)**2+np.arange(1,degree), degree*(degree+1)**2+np.arange(2*degree+1,degree*(degree+1),degree+1),
						  degree*(degree+1)**2+np.arange((degree+1)**2-2,degree*(degree+1),-1),
						  degree*(degree+1)**2+np.arange((degree-1)*(degree+1),degree,-(degree+1))])

	BDindex_F = np.array([np.arange(0,(degree+1)**2),
		(np.matlib.repmat(np.arange(0,degree+1),degree+1,1) + np.matlib.repmat(np.arange(0,degree*(degree+1)**2+1,(degree+1)**2),degree+1,1).transpose()).flatten(),
		(np.matlib.repmat(np.arange(degree,(degree+1)**2,(degree+1)),degree+1,1) + np.matlib.repmat(np.arange(0,degree*(degree+1)**2+1,(degree+1)**2),degree+1,1).transpose()).flatten(),
		(np.matlib.repmat(np.arange((degree+1)**2-1,degree*(degree+1)-1,-1),degree+1,1) + np.matlib.repmat(np.arange(0,degree*(degree+1)**2+1,(degree+1)**2),degree+1,1).transpose()).flatten(),
		(np.matlib.repmat(np.arange(degree*(degree+1),-1,-(degree+1)),degree+1,1) + np.matlib.repmat(np.arange(0,degree*(degree+1)**2+1,(degree+1)**2),degree+1,1).transpose()).flatten(),
		degree*(degree+1)**2+np.arange(0,(degree+1)**2)])

	Iindex = np.setdiff1d(np.arange(0,nru),BDindex_F.flatten())
	Iindex_F = np.setdiff1d(np.arange(0,(degree+1)**2), \
		np.array([np.arange(0,degree+1), np.arange(degree,(degree+1)**2,degree+1), np.arange(0,degree*(degree+1)+1,degree+1), np.arange(degree*(degree+1),(degree+1)**2)]))
	tmp = (np.matlib.reshape(np.arange(degree-2,-1,-1),degree-1,1) + np.matlib.repmat(np.arange(0,nrf,degree-1),degree-1,1).transpose()).flatten()
	faceNr2 = np.zeros(6*nrElems, dtype = int)
	faceNr2[n4fInd]=1
	F_ind = np.reshape(faceNr2,(nrElems,6))
	F_ind = F_ind[:,1:5]
	F_ind = (np.outer(np.arange(0,nrf),F_ind) + np.outer(tmp,1-F_ind)).flatten('F')
	F_ind = np.reshape(F_ind, (nrElems, 4*nrf))

	ind4e = np.zeros((nrElems,nru), dtype = int)
	ind4e[:,[0, degree, (degree+1)**2-1, degree*(degree+1), degree*(degree+1)**2, degree*(degree+1)**2+degree, degree*(degree+1)**2+(degree+1)**2-1, degree*(degree+1)**2+degree*(degree+1)]] = n4e
	ind4e[:,BDindex_S[0,:]] = np.matlib.repmat(nrNodes+s4e[:,0]*nrs,degree-1,1).transpose() + S_ind[:,0:nrs]
	ind4e[:,BDindex_S[1,:]] = np.matlib.repmat(nrNodes+s4e[:,1]*nrs,degree-1,1).transpose() + S_ind[:,nrs:2*nrs]
	ind4e[:,BDindex_S[2,:]] = np.matlib.repmat(nrNodes+s4e[:,2]*nrs,degree-1,1).transpose() + S_ind[:,2*nrs:3*nrs]
	ind4e[:,BDindex_S[3,:]] = np.matlib.repmat(nrNodes+s4e[:,3]*nrs,degree-1,1).transpose() + S_ind[:,3*nrs:4*nrs]
	ind4e[:,BDindex_S[4,:]] = np.matlib.repmat(nrNodes+s4e[:,4]*nrs,degree-1,1).transpose() + S_ind[:,4*nrs:5*nrs]
	ind4e[:,BDindex_S[5,:]] = np.matlib.repmat(nrNodes+s4e[:,5]*nrs,degree-1,1).transpose() + S_ind[:,5*nrs:6*nrs]
	ind4e[:,BDindex_S[6,:]] = np.matlib.repmat(nrNodes+s4e[:,6]*nrs,degree-1,1).transpose() + S_ind[:,6*nrs:7*nrs]
	ind4e[:,BDindex_S[7,:]] = np.matlib.repmat(nrNodes+s4e[:,7]*nrs,degree-1,1).transpose() + S_ind[:,7*nrs:8*nrs]
	ind4e[:,BDindex_S[8,:]] = np.matlib.repmat(nrNodes+s4e[:,8]*nrs,degree-1,1).transpose() + S_ind[:,8*nrs:9*nrs]
	ind4e[:,BDindex_S[9,:]] = np.matlib.repmat(nrNodes+s4e[:,9]*nrs,degree-1,1).transpose() + S_ind[:,9*nrs:10*nrs]
	ind4e[:,BDindex_S[10,:]] = np.matlib.repmat(nrNodes+s4e[:,10]*nrs,degree-1,1).transpose() + S_ind[:,10*nrs:11*nrs]
	ind4e[:,BDindex_S[11,:]] = np.matlib.repmat(nrNodes+s4e[:,11]*nrs,degree-1,1).transpose() + S_ind[:,11*nrs:12*nrs]
	ind4e[:,BDindex_F[0,Iindex_F]] = np.matlib.repmat(nNS + f4e[:,0]*nrf,nrf,1).transpose() + np.matlib.repmat(np.arange(0,nrf),nrElems,1)
	ind4e[:,BDindex_F[1,Iindex_F]] = np.matlib.repmat(nNS + f4e[:,1]*nrf,nrf,1).transpose() + F_ind[:,0:nrf]
	ind4e[:,BDindex_F[2,Iindex_F]] = np.matlib.repmat(nNS + f4e[:,2]*nrf,nrf,1).transpose() + F_ind[:,nrf:2*nrf]
	ind4e[:,BDindex_F[3,Iindex_F]] = np.matlib.repmat(nNS + f4e[:,3]*nrf,nrf,1).transpose() + F_ind[:,2*nrf:3*nrf]
	ind4e[:,BDindex_F[4,Iindex_F]] = np.matlib.repmat(nNS + f4e[:,4]*nrf,nrf,1).transpose() + F_ind[:,3*nrf:4*nrf]
	ind4e[:,BDindex_F[5,Iindex_F]] = np.matlib.repmat(nNS + f4e[:,5]*nrf,nrf,1).transpose() + np.matlib.repmat(np.arange(0,nrf),nrElems,1)
	ind4e[:,Iindex] = np.reshape(np.arange(nNSF,AllNodes),(nrElems,nri))

	ind4Db = np.zeros((n4fDb.shape[0],(degree+1)**2), dtype=int)
	DbF_ind = np.arange(0,n4f.shape[0])
	DbF_ind = DbF_ind[e4f[:,1]==-1]
	DbF = n4f[DbF_ind,:]

	for j in range(0,n4fDb.shape[0]):
		faceDb = DbF_ind[(DbF[:,0] == n4fDb[j,0]) & (DbF[:,1] == n4fDb[j,1]) & (DbF[:,2] == n4fDb[j,2]) & (DbF[:,3] == n4fDb[j,3])]
		elem = e4f[faceDb,0]
		ind4Db[j,:] = ind4e[elem,BDindex_F[f4e[elem,:].flatten()==faceDb,:]]
	ind4Db = np.unique(ind4Db)

	if n4fNb.shape[0] == 0:
		ind4nb = np.zeros(0)
	else:
		ind4Nb = np.zeros((n4fNb.shape[0],(degree+1)**2), dtype=int)
		for j in range(0,n4fNb.shape[0]):
			faceNb = DbF_ind[(DbF[:,0] == n4fNb[j,0]) & (DbF[:,1] == n4fNb[j,1]) & (DbF[:,2] == n4fNb[j,2]) & (DbF[:,3] == n4fNb[j,3])]
			elem = e4f[faceNb,0]
			ind4Nb[j,:] = ind4e[elem,BDindex_F[f4e[elem,:].flatten()==faceNb,:]]

	r, s = RefNodes_Rec(degree)
	r3, s3, t3 = RefNodes_Cube(degree)

	if degree < 2:
		c4nNew = c4n
	else:
		r1D = np.linspace(-1,1,degree+1)
		r1D = r1D[1:-1]
		Mid = (c4n[n4s[:,0],:] + c4n[n4s[:,1],:])/2.0
		c4sx = (Mid[:,0] + np.outer(r1D,(c4n[n4s[:,1],0]-c4n[n4s[:,0],0])/2.0)).flatten('F')
		c4sy = (Mid[:,1] + np.outer(r1D,(c4n[n4s[:,1],1]-c4n[n4s[:,0],1])/2.0)).flatten('F')
		c4sz = (Mid[:,2] + np.outer(r1D,(c4n[n4s[:,1],2]-c4n[n4s[:,0],2])/2.0)).flatten('F')
		c4s = np.array([c4sx, c4sy, c4sz]).transpose().reshape((-1,3))

		c4fx = (np.matlib.repmat(c4n[n4f[:,0],0],nrf,1) + np.outer((r[Iindex_F]+1),(c4n[n4f[:,1],0]-c4n[n4f[:,0],0])/2.0) \
				      + np.outer((s[Iindex_F]+1),(c4n[n4f[:,3],0]-c4n[n4f[:,0],0])/2.0)).flatten('F')
		c4fy = (np.matlib.repmat(c4n[n4f[:,0],1],nrf,1) + np.outer((r[Iindex_F]+1),(c4n[n4f[:,1],1]-c4n[n4f[:,0],1])/2.0) \
				      + np.outer((s[Iindex_F]+1),(c4n[n4f[:,3],1]-c4n[n4f[:,0],1])/2.0)).flatten('F')
		c4fz = (np.matlib.repmat(c4n[n4f[:,0],2],nrf,1) + np.outer((r[Iindex_F]+1),(c4n[n4f[:,1],2]-c4n[n4f[:,0],2])/2.0) \
				      + np.outer((s[Iindex_F]+1),(c4n[n4f[:,3],2]-c4n[n4f[:,0],2])/2.0)).flatten('F')
		c4f = np.array([c4fx, c4fy, c4fz]).transpose().reshape((-1,3))

		c4ix =  (np.matlib.repmat(c4n[n4e[:,0],0],nri,1) \
					  + np.outer((r3[Iindex]+1),(c4n[n4e[:,1],0]-c4n[n4e[:,0],0])/2.0) \
        			  + np.outer((s3[Iindex]+1),(c4n[n4e[:,3],0]-c4n[n4e[:,0],0])/2.0) \
        			  + np.outer((t3[Iindex]+1),(c4n[n4e[:,4],0]-c4n[n4e[:,0],0])/2.0)).flatten('F')
		c4iy =  (np.matlib.repmat(c4n[n4e[:,0],1],nri,1) \
					  + np.outer((r3[Iindex]+1),(c4n[n4e[:,1],1]-c4n[n4e[:,0],1])/2.0) \
        			  + np.outer((s3[Iindex]+1),(c4n[n4e[:,3],1]-c4n[n4e[:,0],1])/2.0) \
        			  + np.outer((t3[Iindex]+1),(c4n[n4e[:,4],1]-c4n[n4e[:,0],1])/2.0)).flatten('F')
		c4iz =  (np.matlib.repmat(c4n[n4e[:,0],2],nri,1) \
					  + np.outer((r3[Iindex]+1),(c4n[n4e[:,1],2]-c4n[n4e[:,0],2])/2.0) \
        			  + np.outer((s3[Iindex]+1),(c4n[n4e[:,3],2]-c4n[n4e[:,0],2])/2.0) \
        			  + np.outer((t3[Iindex]+1),(c4n[n4e[:,4],2]-c4n[n4e[:,0],2])/2.0)).flatten('F')
		c4i = np.array([c4ix, c4iy, c4iz]).transpose().reshape((-1,3))

		c4nNew = np.vstack((c4n, c4s))
		c4nNew = np.vstack((c4nNew, c4f))
		c4nNew = np.vstack((c4nNew, c4i))

	return (c4nNew, ind4e, ind4Db, ind4Nb)

def solve(c4nNew, n4e, ind4e, ind4Db, ind4Nb, M_R, Srr_R, Sss_R, Stt_R, M2D_R, f, u_D, u_N, degree):
	"""
	Refine a given mesh uniformly using the red refinement

	Paramters
		- ``c4nNew`` (``float64 array``) : coordinates for all nodes
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``ind4e`` (``int32 array``) : indices on each element
		- ``ind4Db`` (``int32 array``) : indices on Dirichlet boundary
		- ``ind4Nb`` (``int32 array``) : indices on Neumann boundary
		- ``M_R`` (``float64 array``) : Mass matrix on the reference triangle
		- ``Srr_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_r phi_i \partial_r phi_j dr)
		- ``Sss_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_s phi_i \partial_s phi_j dr)
		- ``Stt_R`` (``float64 array``) : Stiffness matrix on the reference triangle (int_T \partial_t phi_i \partial_t phi_j dr)
		- ``M2D_R`` (``float64 array``) : Mass matrix on the reference rectangle (2D)
		- ``f`` (``lambda function``) : source
		- ``u_D`` (``lambda function``) : Dirichlet boundary condition
		- ``u_N`` (``lambda function``) : Neumann boundary condition
		- ``degree`` (``int32``) : degree of polynomial

	Returns
		- ``x`` (``float64 array``) : solution

	Example
		>>> N = 3
		>>> c4n = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
							[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]])
		>>> n4e = np.array([[0, 1, 3, 2, 4, 5, 7, 6]])
		>>> n4fDb = np.array([[0, 1, 3, 2], [0, 1, 5, 4], [2, 0, 4, 6], [4, 5, 7, 6]])
		>>> n4fNb = np.array([[1, 3, 7, 5]])
		>>> c4nNew, ind4e, ind4Db, ind4Nb = getIndex(N, c4n, n4e, n4fDb, n4fNb)
		>>> M_R, M2D_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R = getMatrix(N)
		>>> f = (lambda x, y, z: 3 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))
		>>> u_D = (lambda x, y, z: x * 0)
		>>> u_N = (lambda x, y, z: np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))
		>>> x = solve(c4nNew, n4e, ind4e, ind4Db, ind4Nb, M_R, Srr_R, Sss_R, Stt_R, M2D_R, f, u_D, u_N)
		>>> x
		array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.36673102,  0.36673102,  0.        ,  0.        ,  0.        ,
				0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.06209753,  0.1873742 ,  0.06209753,  0.1873742 ,  0.61071588,
				0.55509583,  0.61071588,  0.55509583,  0.        ,  0.        ,
				0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
				0.        ,  0.67471301,  0.69313957,  0.77159936,  0.81732792,
				0.67471301,  0.69313957,  0.77159936,  0.81732792])
	"""
	from os import listdir
	from scipy.sparse import coo_matrix
	from scipy.sparse.linalg import spsolve
	from scipy import sparse


	nrElems = n4e.shape[0]
	nrLocal = M_R.shape[0]
	nrNodes = c4nNew.shape[0]
	nrNbSide = ind4Nb.shape[0]

	I = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	J = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.int32)
	Alocal = np.zeros((nrElems * nrLocal * nrLocal), dtype=np.float64)
	b = np.zeros(nrNodes)

	f_val = f(c4nNew[ind4e.flatten(),0], c4nNew[ind4e.flatten(),1], c4nNew[ind4e.flatten(),2])
	g_val = u_N(c4nNew[ind4Nb.flatten(),0], c4nNew[ind4Nb.flatten(),1], c4nNew[ind4Nb.flatten(),2])

	Poison_3D = lib['Poisson_3D_Cube'] # need the extern!!
	Poison_3D.argtypes = (c_void_p, c_void_p, c_void_p, c_void_p, c_int,
						c_int, c_void_p, c_void_p,
						c_void_p, c_void_p, c_void_p, 
						c_int, c_int,
						c_void_p, c_void_p,
						c_void_p, c_void_p, c_void_p, c_void_p)
	Poison_3D.restype = None
	Poison_3D(c_void_p(n4e.ctypes.data), c_void_p(ind4e.ctypes.data),
		c_void_p(ind4Nb.ctypes.data), c_void_p(c4nNew.ctypes.data), c_int(nrElems),
		c_int(nrNbSide), c_void_p(M_R.ctypes.data), c_void_p(M2D_R.ctypes.data),
		c_void_p(Srr_R.ctypes.data), c_void_p(Sss_R.ctypes.data), c_void_p(Stt_R.ctypes.data),
		c_int(nrLocal), c_int(degree),
		c_void_p(f_val.ctypes.data), c_void_p(g_val.ctypes.data),
		c_void_p(I.ctypes.data), c_void_p(J.ctypes.data),
		c_void_p(Alocal.ctypes.data), c_void_p(b.ctypes.data))

	STIMA_COO = coo_matrix((Alocal, (I, J)), shape=(nrNodes, nrNodes))
	STIMA_CSR = STIMA_COO.tocsr()

	dof = np.setdiff1d(range(0,nrNodes), ind4Db)

	x = np.zeros(nrNodes)
	x[ind4Db] = u_D(c4nNew[ind4Db,0], c4nNew[ind4Db,1], c4nNew[ind4Db,2])
	b = b - sparse.csr_matrix.dot(STIMA_CSR,x)
	x[dof] = spsolve(STIMA_CSR[dof, :].tocsc()[:, dof].tocsr(), b[dof])
	
	return x


def computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, exact_uz, approx_u, degree, degree_i):
	"""
	Compute semi H^1-error between exact solution and approximate solution.

	Parameters
		- ``c4n`` (``float64 array``) : coordinates for nodes
		- ``ind4e`` (``int32 array``) : indices for elements
		- ``n4e`` (``int32 array``) : nodes for elements
		- ``exact_u`` (``lambda``) : exact solution
		- ``exact_ux`` (``lambda``) : derivative of exact solution
		- ``exact_uy`` (``lambda``) : derivative of exact solution
		- ``exact_uz`` (``lambda``) : derivative of exact solution
		- ``approx_u`` (``float64 array``) : approximate solution
		- ``degree`` (``int32``) : polynomial degree
		- ``degree_i`` (``int32``) : polynomial degree for interpolation

	Returns
		- ``sH1error`` (``float64``) : semi H^1 error between exact solution and approximate solution.

	Example
		>>> from mozart.mesh.cube import cube
		>>> c4n, ind4e, n4e, n4Db = cube(0,1,0,1,0,1,4,4,4,1)
		>>> f = lambda x,y,z: 3.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		>>> u_D = lambda x,y,z: 0*x
		>>> from mozart.poisson.fem.cube import solve
		>>> x = solve(c4n, ind4e, n4e, n4Db, f, u_D, 1)
		>>> from mozart.poisson.fem.cube import computeError
		>>> exact_u = lambda x,y,z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		>>> exact_ux = lambda x,y,z: np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		>>> exact_uy = lambda x,y,z: np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
		>>> exact_uz = lambda x,y,z: np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)
		>>> sH1error = computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, exact_uz, approx_u, 1, 3)
		>>> sH1error
		0.38357333319
	"""
	# L2error = 0
	sH1error = 0

	M_R, M2D_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R = getMatrix(degree)

	# r = np.linspace(-1, 1, degree + 1)
	# V = VandermondeM1D(degree, r)
	# Dr = Dmatrix1D(degree, r, V)

	# r_i = np.linspace(-1, 1, degree_i + 1)
	# V_i = VandermondeM1D(degree_i, r_i)
	# invV_i = np.linalg.inv(V_i)
	# M_R = np.dot(np.transpose(invV_i), invV_i)
	# PM = VandermondeM1D(degree, r_i)
	# interpM = np.transpose(np.linalg.solve(np.transpose(V), np.transpose(PM)))

	for j in range(0,n4e.shape[0]):
		xr = (c4n[n4e[j,1],0] - c4n[n4e[j,0],0]) / 2.0
		ys = (c4n[n4e[j,3],1] - c4n[n4e[j,0],1]) / 2.0
		zt = (c4n[n4e[j,4],2] - c4n[n4e[j,0],2]) / 2.0
		Jacobi = xr * ys * zt
		rx = 1.0 / xr
		sy = 1.0 / ys
		tz = 1.0 / zt

		Dex = exact_ux(c4n[ind4e[j],0],c4n[ind4e[j],1],c4n[ind4e[j],2]) - rx*np.dot(Dr_R,approx_u[ind4e[j]])
		Dey = exact_uy(c4n[ind4e[j],0],c4n[ind4e[j],1],c4n[ind4e[j],2]) - sy*np.dot(Ds_R,approx_u[ind4e[j]])
		Dez = exact_uz(c4n[ind4e[j],0],c4n[ind4e[j],1],c4n[ind4e[j],2]) - tz*np.dot(Dt_R,approx_u[ind4e[j]])
		sH1error += Jacobi*(np.dot(np.dot(np.transpose(Dex),M_R),Dex) + \
							np.dot(np.dot(np.transpose(Dey),M_R),Dey) + \
							np.dot(np.dot(np.transpose(Dez),M_R),Dez))

	sH1error = np.sqrt(sH1error)
	return sH1error

def getMatrix(degree):
	r = np.linspace(-1, 1, degree+1)
	V = VandermondeM1D(degree, r)
	invV = np.linalg.inv(V)
	M1_R = np.dot(np.transpose(invV),invV)

	M_R = np.kron(np.kron(M1_R,M1_R),M1_R)
	M2D_R = np.kron(M1_R,M1_R)

	Dr_R = np.kron(np.eye(r.size),np.kron(np.eye(r.size),Dmatrix1D(degree, r, V)))
	Ds_R = np.kron(np.eye(r.size),np.kron(Dmatrix1D(degree, r, V),np.eye(r.size)))
	Dt_R = np.kron(Dmatrix1D(degree, r, V),np.kron(np.eye(r.size),np.eye(r.size)))

	Srr_R = np.dot(np.dot(np.transpose(Dr_R),M_R),Dr_R)
	Sss_R = np.dot(np.dot(np.transpose(Ds_R),M_R),Ds_R)
	Stt_R = np.dot(np.dot(np.transpose(Dt_R),M_R),Dt_R)

	return (M_R, M2D_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R)