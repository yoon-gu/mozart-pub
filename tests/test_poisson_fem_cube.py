import unittest
import numpy as np
from numpy import linalg as LA

class TestFemCube(unittest.TestCase):
	def test_getPoissonMatrix3D(self):
		from mozart.poisson.fem.cube import getMatrix
		M_R, Srr_R, Sss_R, Stt_R, Dr_R, Ds_R, Dt_R = getMatrix(1)
		diff_M_R = M_R - np.array([[8, 4, 4, 2, 4, 2, 2, 1],
							       [4, 8, 2, 4, 2, 4, 1, 2],
							       [4, 2, 8, 4, 2, 1, 4, 2],
							       [2, 4, 4, 8, 1, 2, 2, 4],
							       [4, 2, 2, 1, 8, 4, 4, 2],
							       [2, 4, 1, 2, 4, 8, 2, 4],
							       [2, 1, 4, 2, 4, 2, 8, 4],
							       [1, 2, 2, 4, 2, 4, 4, 8]]) / 27.
		diff_Srr_R = Srr_R - np.array([[ 4, -4,  2, -2,  2, -2,  1, -1],
									   [-4,  4, -2,  2, -2,  2, -1,  1],
									   [ 2, -2,  4, -4,  1, -1,  2, -2],
									   [-2,  2, -4,  4, -1,  1, -2,  2],
									   [ 2, -2,  1, -1,  4, -4,  2, -2],
									   [-2,  2, -1,  1, -4,  4, -2,  2],
									   [ 1, -1,  2, -2,  2, -2,  4, -4],
									   [-1,  1, -2,  2, -2,  2, -4,  4 ]]) / 18.
		diff_Sss_R = Sss_R - np.array([[ 4,  2, -4, -2,  2,  1, -2, -1],
									   [ 2,  4, -2, -4,  1,  2, -1, -2],
									   [-4, -2,  4,  2, -2, -1,  2,  1],
									   [-2, -4,  2,  4, -1, -2,  1,  2],
									   [ 2,  1, -2, -1,  4,  2, -4, -2],
									   [ 1,  2, -1, -2,  2,  4, -2, -4],
									   [-2, -1,  2,  1, -4, -2,  4,  2],
									   [-1, -2,  1,  2, -2, -4,  2,  4]]) / 18.
		diff_Stt_R = Stt_R - np.array([[ 4,  2,  2,  1, -4, -2, -2, -1],
									   [ 2,  4,  1,  2, -2, -4, -1, -2],
									   [ 2,  1,  4,  2, -2, -1, -4, -2],
									   [ 1,  2,  2,  4, -1, -2, -2, -4],
									   [-4, -2, -2, -1,  4,  2,  2,  1],
									   [-2, -4, -1, -2,  2,  4,  1,  2],
									   [-2, -1, -4, -2,  2,  1,  4,  2],
									   [-1, -2, -2, -4,  1,  2,  2,  4]]) / 18.

		diff_Dr_R = Dr_R - np.array([[-1, 1,  0, 0,  0, 0,  0, 0],
								     [-1, 1,  0, 0,  0, 0,  0, 0],
								     [ 0, 0, -1, 1,  0, 0,  0, 0],
								     [ 0, 0, -1, 1,  0, 0,  0, 0],
								     [ 0, 0,  0, 0, -1, 1,  0, 0],
								     [ 0, 0,  0, 0, -1, 1,  0, 0],
								     [ 0, 0,  0, 0,  0, 0, -1, 1],
								     [ 0, 0,  0, 0,  0, 0, -1, 1]]) / 2.
		diff_Ds_R = Ds_R - np.array([[-1,  0, 1, 0,  0,  0, 0, 0],
									 [ 0, -1, 0, 1,  0,  0, 0, 0],
									 [-1,  0, 1, 0,  0,  0, 0, 0],
									 [ 0, -1, 0, 1,  0,  0, 0, 0],
									 [ 0,  0, 0, 0, -1,  0, 1, 0],
									 [ 0,  0, 0, 0,  0, -1, 0, 1],
									 [ 0,  0, 0, 0, -1,  0, 1, 0],
									 [ 0,  0, 0, 0,  0, -1, 0, 1]]) / 2.
		diff_Dt_R = Dt_R - np.array([[-1,  0,  0,  0, 1, 0, 0, 0],
									 [ 0, -1,  0,  0, 0, 1, 0, 0],
									 [ 0,  0, -1,  0, 0, 0, 1, 0],
									 [ 0,  0,  0, -1, 0, 0, 0, 1],
									 [-1,  0,  0,  0, 1, 0, 0, 0],
									 [ 0, -1,  0,  0, 0, 1, 0, 0],
									 [ 0,  0, -1,  0, 0, 0, 1, 0],
									 [ 0,  0,  0, -1, 0, 0, 0, 1]]) / 2.

		self.assertAlmostEqual(LA.norm(diff_M_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Srr_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Sss_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Stt_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Dr_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Ds_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Dt_R), 0.0, 8)

	def test_solve(self):
		from mozart.mesh.cube import cube
		from mozart.poisson.fem.cube import solve
		x1, x2, y1, y2, z1, z2, Mx, My, Mz, N = (0, 1, 0, 1, 0, 1, 3, 3, 3, 1)
		c4n, ind4e, n4e, n4Db = cube(x1,x2,y1,y2,z1,z2,Mx,My,Mz,N)
		f = lambda x,y,z: 3.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
		u_D = lambda x,y,z: 0*x
		x = solve(c4n, ind4e, n4e, n4Db, f, u_D, N)
		diff_x = x - np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   							   0.593564453933756, 0.593564453933756, 0, 0, 0.593564453933756,
							   0.593564453933756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
							   0.593564453933756, 0.593564453933756, 0, 0, 0.593564453933756,
   							   0.593564453933756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

		self.assertAlmostEqual(LA.norm(diff_x), 0.0, 8)

	def test_computeError(self):
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
			rateH1=(np.log(sH1error[1:])-np.log(sH1error[0:-1]))/(np.log(h[1:])-np.log(h[0:-1]))
			self.assertTrue(np.abs(rateH1[-1]) > degree-0.2, \
				"Convergence rate : {0} when trying degree = {1}".format(np.abs(rateH1[-1]), degree))
