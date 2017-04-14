import unittest
import numpy as np
from numpy import linalg as LA

class TestFemRectangle(unittest.TestCase):
	def test_2d_uniform_rectangle(self):
		from mozart.mesh.rectangle import rectangle
		x1, x2, y1, y2, Mx, My, N = (0, 1, 0, 1, 2, 2, 1)
		c4n, ind4e, n4e, n4db = rectangle(x1,x2,y1,y2,Mx,My,N)
		diff_c4n = c4n - np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],[0.0,0.5],[0.5,0.5],[1.0,0.5],[0.0,1.0],[0.5,1.0],[1.0,1.0]])
		diff_ind4e = ind4e - np.array([[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]])
		diff_n4e = n4e - np.array([[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]])
		diff_n4db = n4db - np.array([0,1,2,3,5,6,7,8])

		self.assertAlmostEqual(LA.norm(diff_c4n), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_n4e), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_n4db), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_ind4e), 0.0, 8)