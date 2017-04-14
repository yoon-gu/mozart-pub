import unittest
import numpy as np
from numpy import linalg as LA

class TestFemInterval(unittest.TestCase):
	def test_1d_uniform_interval(self):
		from mozart.mesh.interval import interval
		a, b, M, N = (0, 1, 4, 2)
		c4n, n4e, n4db, ind4e = interval(a,b,M,N)
		diff_c4n = c4n - np.linspace(a,b,M*N+1)
		diff_n4e = n4e - np.array([[0,2], [2,4], [4,6], [6,8]])
		diff_n4db = n4db - np.array([0, 8])
		diff_ind4e = ind4e - np.array([[0,1,2], [2,3,4], [4,5,6], [6,7,8]])
		self.assertTrue(LA.norm(diff_c4n) < 1E-8)
		self.assertTrue(LA.norm(diff_n4e) < 1E-8)
		self.assertTrue(LA.norm(diff_n4db) < 1E-8)
		self.assertTrue(LA.norm(diff_ind4e) < 1E-8)

	def test_getPoissonMatrix1D(self):
		from mozart.poisson.fem.interval import getMatrix
		M1, S1, D1 = getMatrix(1)
		diff_M1 = M1 - np.array([[ 2,  1], [ 1, 2]]) / 3.
		diff_S1 = S1 - np.array([[ 1, -1], [-1, 1]]) / 2.
		diff_D1 = D1 - np.array([[-1,  1], [-1, 1]]) / 2.
		M2, S2, D2 = getMatrix(2)
		diff_M2 = M2 - np.array([[ 4,  2, -1], [ 2, 16,  2], [-1,  2, 4]]) / 15.
		diff_S2 = S2 - np.array([[ 7, -8,  1], [-8, 16, -8], [ 1, -8, 7]]) / 6.
		diff_D2 = D2 - np.array([[-3,  4, -1], [-1,  0,  1], [ 1, -4, 3]]) / 2.
		self.assertTrue(LA.norm(diff_M1) < 1E-8)
		self.assertTrue(LA.norm(diff_S1) < 1E-8)
		self.assertTrue(LA.norm(diff_D1) < 1E-8)
		self.assertTrue(LA.norm(diff_M2) < 1E-8)
		self.assertTrue(LA.norm(diff_S2) < 1E-8)
		self.assertTrue(LA.norm(diff_D2) < 1E-8)

	def test_solve(self):
		from mozart.mesh.interval import interval
		N = 3
		c4n, n4e, n4db, ind4e = interval(0, 1, 4, N)
		f = lambda x: np.ones_like(x)
		u_D = lambda x: np.zeros_like(x)
		from mozart.poisson.fem.interval import solve
		x = solve(c4n, n4e, n4db, ind4e, f, u_D, N)
		diff_x = x - np.array([                 0,   0.038194444444444,   0.069444444444444,   0.093749999999999,   0.111111111111110,
		   0.121527777777777,   0.124999999999999,   0.121527777777777,   0.111111111111110,   0.093749999999999,   0.069444444444444,
		   0.038194444444444,                   0])
		self.assertTrue(LA.norm(diff_x) < 1E-8)

	def test_computeError(self):
		from mozart.mesh.interval import interval
		from mozart.poisson.fem.interval import solve
		from mozart.poisson.fem.interval import computeError
		iter = 4
		for N in range(1,4):
			f = lambda x: np.pi ** 2 * np.sin(np.pi * x)
			u_D = lambda x: np.zeros_like(x)
			exact_u = lambda x: np.sin(np.pi * x)
			exact_ux = lambda x: np.pi * np.cos(np.pi * x)
			L2error = np.zeros(iter, dtype = np.float64)
			sH1error = np.zeros(iter, dtype = np.float64)
			h = np.zeros(iter, dtype = np.float64)
			for j in range(0,iter):
				c4n, n4e, n4db, ind4e = interval(0,1,2**(j+1),N)
				x = solve(c4n, n4e, n4db, ind4e, f, u_D, N)
				L2error[j], sH1error[j] = computeError(c4n, n4e, ind4e, exact_u, exact_ux, x, N, N+3)
				h[j] = 1 / 2.0**(j+1)
			rateL2=(np.log(L2error[1:])-np.log(L2error[0:-1]))/(np.log(h[1:])-np.log(h[0:-1]))
			rateH1=(np.log(sH1error[1:])-np.log(sH1error[0:-1]))/(np.log(h[1:])-np.log(h[0:-1]))
			self.assertTrue(np.abs(rateL2[-1]) > N+0.9)
			self.assertTrue(np.abs(rateH1[-1]) > N-0.1)
