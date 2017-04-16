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
		self.assertAlmostEqual(LA.norm(diff_c4n), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_n4e), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_n4db), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_ind4e), 0.0, 8)