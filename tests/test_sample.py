import unittest
import numpy as np
from numpy import linalg as LA
class TestMethods(unittest.TestCase):

	def test_test(self):
		self.assertTrue(True)

	def test_import(self):
		import mozart as mz
		self.assertTrue(True)

	def test_authors(self):
		import mozart as mz
		authors = ('Yoon-gu Hwang <yz0624@gmail.com>', 'Dong-Wook Shin <dwshin.yonsei@gmail.com>', 'Ji-Yeon Suh <suh91919@gmail.com>')
		self.assertEqual(mz.__author__, authors)
	def test_1d_uniform_mesh(self):
		from mozart.mesh.rectangle import unit_interval
		N = 4
		c4n, n4e = unit_interval(N)
		diff_c4n = c4n - np.linspace(0,1,N)
		diff_n4e = n4e - np.array([[0,1], [1, 2], [2,3]])
		self.assertTrue(LA.norm(diff_c4n) < 1E-8)
		self.assertTrue(LA.norm(diff_n4e) < 1E-8)

	def test_poisson_square_2d(self):
		from mozart.mesh.rectangle import unit_square
		unit_square(0.1)
		self.assertTrue(True)

	def test_getPoissonMatrix1D(self):
		from mozart.poisson.solve import getMatrix1D
		for k in range(1,3):
			getMatrix1D(k)
		self.assertTrue(True)

	def test_nJacobiP(self):
		from mozart.poisson.solve import nJacobiP
		for k in range(1,5):
			x = np.linspace(-1,1,k+1)
			nJacobiP(x,0,0,k)
		self.assertTrue(True)

	def test_solve_onedim(self):
		from mozart.mesh.rectangle import unit_interval
		N = 3
		c4n, n4e = unit_interval(N)
		n4Db = [0, N-1]
		f = lambda x: np.ones_like(x)
		u_D = lambda x: np.zeros_like(x)
		from mozart.poisson.solve import one_dim
		x = one_dim(c4n, n4e, n4Db, f, u_D)
		diff_x = x - np.array([0., 0.125, 0.])
		self.assertTrue(LA.norm(diff_x) < 1E-8)

		self.assertTrue(True)

	def test_solve_twodim(self):
		from mozart.poisson.solve import two_dim
		two_dim(None, None, None, None)

		self.assertTrue(True)

	def test_solve_threedim(self):
		from mozart.poisson.solve import three_dim
		three_dim(None, None, None, None)

		self.assertTrue(True)

	def test_benchmark01_sample(self):
		from mozart.poisson.solve import sample
		sample()
		self.assertTrue(True)