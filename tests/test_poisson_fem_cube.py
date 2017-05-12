import numpy as np
import numpy.testing as npt
from numpy import linalg as LA

def test_compute_n4s():
	from mozart.poisson.fem.cube import compute_n4s
	n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
	n4s = compute_n4s(n4e)
	ref_n4s = np.array([[ 0,  1], [ 1,  2], [ 1,  4], [ 2,  5], [ 4,  3], 
		[ 5,  4], [ 3,  0], [ 0,  6], [ 1,  7], [ 2,  8], [ 4, 10], 
		[ 5, 11], [ 3,  9], [ 6,  7], [ 7,  8], [ 7, 10], [ 8, 11], 
		[10,  9], [11, 10], [ 9,  6]], dtype = int)
	npt.assert_almost_equal(n4s, ref_n4s, decimal=8)

def test_compute_s4e():
	from mozart.poisson.fem.cube import compute_s4e
	n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
	s4e = compute_s4e(n4e)
	ref_s4e = np.array([[ 0,  2,  4,  6,  7,  8, 10, 12, 13, 15, 17, 19],
				[ 1,  3,  5,  2,  8,  9, 11, 10, 14, 16, 18, 15]], dtype = int)
	npt.assert_almost_equal(s4e, ref_s4e, decimal=8)

def test_compute_n4f():
	from mozart.poisson.fem.cube import compute_n4f
	n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
	n4f = compute_n4f(n4e)
	ref_n4f = np.array([[ 0,  1,  4,  3], [ 1,  2,  5,  4], [ 0,  1,  7,  6], [ 1,  2,  8,  7], 
				[ 1,  4, 10,  7], [ 2,  5, 11,  8], [ 4,  3,  9, 10], [ 5,  4, 10, 11], 
				[ 3,  0,  6,  9], [ 6,  7, 10,  9], [ 7,  8, 11, 10]], dtype = int)
	npt.assert_almost_equal(n4f, ref_n4f, decimal=8)

def test_compute_f4e():
	from mozart.poisson.fem.cube import compute_f4e
	n4e = np.array([[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]])
	f4e = compute_f4e(n4e)
	ref_f4e = np.array([[ 0,  2,  4,  6,  8,  9],
				[ 1,  3,  5,  7,  4, 10]])
	npt.assert_almost_equal(f4e, ref_f4e, decimal=8)

def test_getPoissonMatrix3D():
	from mozart.poisson.fem.cube import getMatrix
	M_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R = getMatrix(1)
	ref_M_R = np.array([[8, 4, 4, 2, 4, 2, 2, 1],
				       [4, 8, 2, 4, 2, 4, 1, 2],
				       [4, 2, 8, 4, 2, 1, 4, 2],
				       [2, 4, 4, 8, 1, 2, 2, 4],
				       [4, 2, 2, 1, 8, 4, 4, 2],
				       [2, 4, 1, 2, 4, 8, 2, 4],
				       [2, 1, 4, 2, 4, 2, 8, 4],
				       [1, 2, 2, 4, 2, 4, 4, 8]]) / 27.
	ref_Srr_R = np.array([[ 4, -4,  2, -2,  2, -2,  1, -1],
						 [-4,  4, -2,  2, -2,  2, -1,  1],
						 [ 2, -2,  4, -4,  1, -1,  2, -2],
						 [-2,  2, -4,  4, -1,  1, -2,  2],
						 [ 2, -2,  1, -1,  4, -4,  2, -2],
						 [-2,  2, -1,  1, -4,  4, -2,  2],
						 [ 1, -1,  2, -2,  2, -2,  4, -4],
						 [-1,  1, -2,  2, -2,  2, -4,  4 ]]) / 18.
	ref_Sss_R = np.array([[ 4,  2, -4, -2,  2,  1, -2, -1],
						 [ 2,  4, -2, -4,  1,  2, -1, -2],
						 [-4, -2,  4,  2, -2, -1,  2,  1],
						 [-2, -4,  2,  4, -1, -2,  1,  2],
						 [ 2,  1, -2, -1,  4,  2, -4, -2],
						 [ 1,  2, -1, -2,  2,  4, -2, -4],
						 [-2, -1,  2,  1, -4, -2,  4,  2],
						 [-1, -2,  1,  2, -2, -4,  2,  4]]) / 18.
	ref_Stt_R = np.array([[ 4,  2,  2,  1, -4, -2, -2, -1],
						 [ 2,  4,  1,  2, -2, -4, -1, -2],
						 [ 2,  1,  4,  2, -2, -1, -4, -2],
						 [ 1,  2,  2,  4, -1, -2, -2, -4],
						 [-4, -2, -2, -1,  4,  2,  2,  1],
						 [-2, -4, -1, -2,  2,  4,  1,  2],
						 [-2, -1, -4, -2,  2,  1,  4,  2],
						 [-1, -2, -2, -4,  1,  2,  2,  4]]) / 18.
	ref_Dr_R = np.array([[-1, 1,  0, 0,  0, 0,  0, 0],
						[-1, 1,  0, 0,  0, 0,  0, 0],
						[ 0, 0, -1, 1,  0, 0,  0, 0],
						[ 0, 0, -1, 1,  0, 0,  0, 0],
						[ 0, 0,  0, 0, -1, 1,  0, 0],
						[ 0, 0,  0, 0, -1, 1,  0, 0],
						[ 0, 0,  0, 0,  0, 0, -1, 1],
						[ 0, 0,  0, 0,  0, 0, -1, 1]]) / 2.
	ref_Ds_R = np.array([[-1,  0, 1, 0,  0,  0, 0, 0],
						[ 0, -1, 0, 1,  0,  0, 0, 0],
						[-1,  0, 1, 0,  0,  0, 0, 0],
						[ 0, -1, 0, 1,  0,  0, 0, 0],
						[ 0,  0, 0, 0, -1,  0, 1, 0],
						[ 0,  0, 0, 0,  0, -1, 0, 1],
						[ 0,  0, 0, 0, -1,  0, 1, 0],
						[ 0,  0, 0, 0,  0, -1, 0, 1]]) / 2.
	ref_Dt_R = np.array([[-1,  0,  0,  0, 1, 0, 0, 0],
						[ 0, -1,  0,  0, 0, 1, 0, 0],
						[ 0,  0, -1,  0, 0, 0, 1, 0],
						[ 0,  0,  0, -1, 0, 0, 0, 1],
						[-1,  0,  0,  0, 1, 0, 0, 0],
						[ 0, -1,  0,  0, 0, 1, 0, 0],
						[ 0,  0, -1,  0, 0, 0, 1, 0],
						[ 0,  0,  0, -1, 0, 0, 0, 1]]) / 2.
	npt.assert_almost_equal(M_R, ref_M_R, decimal=8)
	npt.assert_almost_equal(Srr_R, ref_Srr_R, decimal=8)
	npt.assert_almost_equal(Sss_R, ref_Sss_R, decimal=8)
	npt.assert_almost_equal(Stt_R, ref_Stt_R, decimal=8)
	npt.assert_almost_equal(Dr_R, ref_Dr_R, decimal=8)
	npt.assert_almost_equal(Ds_R, ref_Ds_R, decimal=8)
	npt.assert_almost_equal(Dt_R, ref_Dt_R, decimal=8)

def test_solve():
	from mozart.mesh.cube import cube
	from mozart.poisson.fem.cube import solve
	x1, x2, y1, y2, z1, z2, Mx, My, Mz, N = (0, 1, 0, 1, 0, 1, 3, 3, 3, 1)
	c4n, ind4e, n4e, n4Db = cube(x1,x2,y1,y2,z1,z2,Mx,My,Mz,N)
	f = lambda x,y,z: 3.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
	u_D = lambda x,y,z: 0*x
	x = solve(c4n, ind4e, n4e, n4Db, f, u_D, N)
	ref_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					   0.593564453933756, 0.593564453933756, 0, 0, 0.593564453933756,
					   0.593564453933756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					   0.593564453933756, 0.593564453933756, 0, 0, 0.593564453933756,
					   0.593564453933756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

	npt.assert_almost_equal(x, ref_x, decimal=8)

def test_computeError():
	from mozart.mesh.cube import cube
	from mozart.poisson.fem.cube import solve
	from mozart.poisson.fem.cube import computeError
	iter = 3
	for degree in range(1,4):
		f = lambda x,y,z: 3.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		u_D = lambda x,y,z: 0*x
		exact_u = lambda x,y,z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		exact_ux = lambda x,y,z: np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		exact_uy = lambda x,y,z: np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)
		exact_uz = lambda x,y,z: np.pi*np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)
		sH1error = np.zeros(iter, dtype = np.float64)
		h = np.zeros(iter, dtype = np.float64)
		for j in range(0,iter):
			c4n, ind4e, n4e, n4Db = cube(0,1,0,1,0,1,2**(j+1),2**(j+1),2**(j+1),degree)
			x = solve(c4n, ind4e, n4e, n4Db, f, u_D, degree)
			sH1error[j] = computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, exact_uz, x, degree, degree+3)
			h[j] = 1 / 2.0**(j+1)
		ratesH1=(np.log(sH1error[1:])-np.log(sH1error[0:-1]))/(np.log(h[1:])-np.log(h[0:-1]))
		npt.assert_array_less(degree-0.2, ratesH1[-1])
		#self.assertTrue(np.abs(rateH1[-1]) > degree-0.2, \
				#"Convergence rate : {0} when trying degree = {1}".format(np.abs(rateH1[-1]), degree))