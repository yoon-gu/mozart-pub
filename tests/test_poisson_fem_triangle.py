import unittest
import numpy as np
from numpy import linalg as LA

class TestFemTriangle(unittest.TestCase):
	def test_getMatrix2D(self):
		from mozart.poisson.fem.triangle import getMatrix
		N = 1
		M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R = getMatrix(N)
		diff_M = M_R - np.array([[1.0/3, 1.0/6, 1.0/6,], [1.0/6, 1.0/3, 1.0/6], [1.0/6, 1.0/6, 1.0/3]], dtype = np.float64)
		diff_Srr = Srr_R - np.array([[1.0/2, -1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0], [0.0, 0.0, 0.0]], dtype = np.float64)
		diff_Srs = Srs_R - np.array([[1.0/2, 0.0, -1.0/2], [-1.0/2, 0.0, 1.0/2], [0.0, 0.0, 0.0]], dtype = np.float64)
		diff_Ssr = Ssr_R - np.array([[1.0/2, -1.0/2, 0.0], [0.0, 0.0, 0.0], [-1.0/2, 1.0/2, 0.0]], dtype = np.float64)
		diff_Sss = Sss_R - np.array([[1.0/2, 0.0, -1.0/2], [0.0, 0.0, 0.0], [-1.0/2, 0.0, 1.0/2]], dtype = np.float64)
		diff_Dr = Dr_R - np.array([[-1.0/2, 1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0]], dtype = np.float64)
		diff_Ds = Ds_R - np.array([[-1.0/2, 0.0, 1.0/2], [-1.0/2, 0.0, 1.0/2], [-1.0/2, 0.0, 1.0/2]], dtype = np.float64)
		self.assertAlmostEqual(LA.norm(diff_M), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Srr), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Srs), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Ssr), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Sss), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Dr), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Ds), 0.0, 8)