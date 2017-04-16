import unittest
import numpy as np
from numpy import linalg as LA

class TestFemRectangle(unittest.TestCase):
	def test_getPoissonMatrix2D(self):
		from mozart.poisson.fem.rectangle import getMatrix
		M_R, Srr_R, Sss_R, Dr_R, Ds_R = getMatrix(1)
		diff_M_R = M_R - np.array([[4, 2, 2, 1],[2, 4, 1, 2],[2, 1, 4, 2],[1, 2 ,2 ,4]]) / 9.
		diff_Srr_R = Srr_R - np.array([[2, -2 ,1 ,-1],[-2 ,2 ,-1, 1],[1 ,-1 ,2 ,-2],[-1, 1, -2 ,2]]) / 6.
		diff_Sss_R = Sss_R - np.array([[2 ,1, -2 ,-1],[1 ,2, -1 ,-2],[-2 ,-1, 2, 1],[-1, -2 ,1 ,2]]) / 6.
		diff_Dr_R = Dr_R - np.array([[-1, 1, 0, 0],[-1, 1, 0, 0],[0 ,0 ,-1, 1],[0, 0, -1 ,1]]) / 2.
		diff_Ds_R = Ds_R - np.array([[-1, 0, 1, 0],[0 ,-1, 0, 1],[-1, 0, 1, 0],[0, -1 ,0 ,1]]) / 2.

		self.assertAlmostEqual(LA.norm(diff_M_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Srr_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Sss_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Dr_R), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Ds_R), 0.0, 8)

	def test_solve(self):
		from mozart.mesh.rectangle import rectangle
		from mozart.poisson.fem.rectangle import solve
		x1, x2, y1, y2, Mx, My, N = (0, 1, 0, 1, 4, 4, 1)
		c4n, ind4e, n4e, n4Db = rectangle(x1,x2,y1,y2,Mx,My,N)
		f = lambda x,y: 2.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
		u_D = lambda x,y: 0*x
		x = solve(c4n, ind4e, n4e, n4Db, f, u_D, N)
		diff_x = x - np.array([0,                   0,                   0,                   0,                   0,
                   			   0,   0.475110454183750,   0.671907647931901,   0.475110454183750,                   0,
                               0,   0.671907647931901,   0.950220908367501,   0.671907647931901,       			   0,
                               0,   0.475110454183750,   0.671907647931901,   0.475110454183750,                   0,
			                   0,                   0,                   0,                   0,                   0])

		self.assertAlmostEqual(LA.norm(diff_x), 0.0, 8)

	def test_computeError(self):
		from mozart.mesh.rectangle import rectangle
		from mozart.poisson.fem.rectangle import solve
		from mozart.poisson.fem.rectangle import computeError
		iter = 3
		for degree in range(1,4):
			f = lambda x,y: 2.0*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
			u_D = lambda x,y: 0*x
			exact_u = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
			exact_ux = lambda x,y: np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
			exact_uy = lambda x,y: np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
			sH1error = np.zeros(iter, dtype = np.float64)
			h = np.zeros(iter, dtype = np.float64)
			for j in range(0,iter):
				c4n, ind4e, n4e, n4Db = rectangle(0,1,0,1,2**(j+1),2**(j+1),degree)
				x = solve(c4n, ind4e, n4e, n4Db, f, u_D, degree)
				sH1error[j] = computeError(c4n, n4e, ind4e, exact_u, exact_ux, exact_uy, x, degree, degree+3)
				h[j] = 1 / 2.0**(j+1)
			rateH1=(np.log(sH1error[1:])-np.log(sH1error[0:-1]))/(np.log(h[1:])-np.log(h[0:-1]))
			self.assertTrue(np.abs(rateH1[-1]) > degree-0.1, \
				"Convergence rate : {0} when trying degree = {1}".format(np.abs(rateH1[-1]), degree))
