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
		self.assertTrue(LA.norm(diff_M) < 1E-8)
		self.assertTrue(LA.norm(diff_Srr) < 1E-8)
		self.assertTrue(LA.norm(diff_Srs) < 1E-8)
		self.assertTrue(LA.norm(diff_Ssr) < 1E-8)
		self.assertTrue(LA.norm(diff_Sss) < 1E-8)
		self.assertTrue(LA.norm(diff_Dr) < 1E-8)
		self.assertTrue(LA.norm(diff_Ds) < 1E-8)

	def test_compute_n4s(self):
		from mozart.mesh.triangle import compute_n4s
		n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
		n4s = compute_n4s(n4e)
		diff_n4s = n4s - np.array([[1, 3], [2, 4], [4, 6], [5, 7], [3, 0], [1, 4], [2, 5],
			[6, 3], [4, 7], [5, 8], [0, 1], [4, 3], [1, 2], [5, 4], [7, 6], [8, 7]])
		self.assertTrue(LA.norm(diff_n4s) < 1E-8)

	def test_compute_s4e(self):
		from mozart.mesh.triangle import compute_s4e
		n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
		s4e = compute_s4e(n4e)
		diff_s4e = s4e - np.array([[0, 4, 10], [0, 5, 11], [1, 5, 12], [1, 6, 13],
			[2, 7, 11], [2, 8, 14], [3, 8, 13], [3, 9, 15]])
		self.assertTrue(LA.norm(diff_s4e) < 1E-8)

	def test_compute_e4s(self):
		from mozart.mesh.triangle import compute_e4s
		n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
		e4s = compute_e4s(n4e)
		diff_e4s = e4s - np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, -1], [1, 2],
			[3, -1], [4, -1], [5, 6], [7, -1], [0, -1], [1, 4], [2, -1], [3, 6], [5, -1], [7, -1]])
		self.assertTrue(LA.norm(diff_e4s) < 1E-8)

	def test_refineUniformRed(self):
		from mozart.mesh.triangle import refineUniformRed
		c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5]])
		n4e = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
		n4Db = np.array([[0, 1], [1, 2]])
		n4Nb = np.array([[2, 3], [3, 0]])
		c4n, n4e, n4Db, n4Nb = refineUniformRed(c4n, n4e, n4Db, n4Nb)
		diff_c4n = c4n - np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5], [0.5, 0.],
			[1., 0.5], [0.5, 1.], [0., 0.5], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]])
		diff_n4e = n4e - np.array([[0, 5, 12], [5, 1, 9], [9, 12, 5], [12, 9, 4], [1, 6, 9], [6, 2, 10], [10, 9, 6],
			[9, 10, 4], [2, 7, 10], [7, 3, 11], [11, 10, 7], [10, 11, 4], [3, 8, 11], [8, 0, 12], [12, 11, 8], [11, 12, 4]])
		diff_n4Db = n4Db - np.array([[0, 5], [5, 1], [1, 6], [6, 2]])
		diff_n4Nb = n4Nb - np.array([[2, 7], [7, 3], [3, 8], [8, 0]])
		self.assertTrue(LA.norm(diff_c4n) < 1E-8)
		self.assertTrue(LA.norm(diff_n4e) < 1E-8)
		self.assertTrue(LA.norm(diff_n4Db) < 1E-8)
		self.assertTrue(LA.norm(diff_n4Nb) < 1E-8)
