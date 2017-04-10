import unittest
import numpy as np
from numpy import linalg as LA
class TestBasic(unittest.TestCase):
	def test_test(self):
		self.assertTrue(True)

	def test_import(self):
		import mozart as mz
		self.assertTrue(True)

	def test_authors(self):
		import mozart as mz
		authors = ('Yoon-gu Hwang <yz0624@gmail.com>', 'Dong-Wook Shin <dwshin.yonsei@gmail.com>', 'Ji-Yeon Suh <suh91919@gmail.com>')
		self.assertEqual(mz.__author__, authors)

class TestFemCommon(unittest.TestCase):
	def test_nJacobiP(self):
		from mozart.poisson.fem.common import nJacobiP
		x = np.linspace(-1,1,5)
		P = nJacobiP(x,0,0,0)
		diff_P = P - 0.707106781186548 * np.ones(5,float)
		P2 = nJacobiP(x,0,0,1)
		diff_P2 = P2 - 1.224744871391589*x
		P3 = nJacobiP(x,0,0,2)
		diff_P3 = P3 - np.array([1.581138830084190, -0.197642353760524, -0.790569415042095, -0.197642353760524, 1.581138830084190])
		self.assertTrue(LA.norm(diff_P) < 1E-8)
		self.assertTrue(LA.norm(diff_P2) < 1E-8)
		self.assertTrue(LA.norm(diff_P3) < 1E-8)

	def test_nJacobiGQ(self):
		from mozart.poisson.fem.common import nJacobiGQ
		x0, w0 = nJacobiGQ(0,0,0)
		diff_x0 = x0 - 0.0
		diff_w0 = w0 - 2.0
		x1, w1 = nJacobiGQ(0,0,1)
		diff_x1 = x1 - np.array([-np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)])
		diff_w1 = w1 - np.array([1.0, 1.0])
		x2, w2 = nJacobiGQ(0,0,2)
		diff_x2 = x2 - np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
		diff_w2 = w2 - np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
		x3, w3 = nJacobiGQ(1,1,1)
		diff_x3 = x3 - np.array([-0.447213595499958,   0.447213595499958])
		diff_w3 = w3 - np.array([0.666666666666667,   0.666666666666667])
		self.assertTrue(LA.norm(diff_x0) < 1E-8)
		self.assertTrue(LA.norm(diff_w0) < 1E-8)
		self.assertTrue(LA.norm(diff_x1) < 1E-8)
		self.assertTrue(LA.norm(diff_w1) < 1E-8)
		self.assertTrue(LA.norm(diff_x2) < 1E-8)
		self.assertTrue(LA.norm(diff_w2) < 1E-8)
		self.assertTrue(LA.norm(diff_x3) < 1E-8)
		self.assertTrue(LA.norm(diff_w3) < 1E-8)

	def test_nJacobiGL(self):
		from mozart.poisson.fem.common import nJacobiGL
		x0 = nJacobiGL(0,0,0)
		diff_x0 = x0 - 0.0;
		x1 = nJacobiGL(0,0,1)
		diff_x1 = x1 - np.array([-1.0, 1.0])
		x2 = nJacobiGL(0,0,2)
		diff_x2 = x2 - np.array([-1.0, 0.0, 1.0])
		self.assertTrue(LA.norm(diff_x0) < 1E-8)
		self.assertTrue(LA.norm(diff_x1) < 1E-8)
		self.assertTrue(LA.norm(diff_x2) < 1E-8)

	def test_DnJacobiP(self):
		from mozart.poisson.fem.common import DnJacobiP
		x = np.linspace(-1,1,5)
		dP = DnJacobiP(x,0,0,0)
		diff_dP = dP - np.zeros(5,float)
		dP2 = DnJacobiP(x,0,0,1)
		diff_dP2 = dP2 - 1.224744871391589*np.ones(5,float)
		self.assertTrue(LA.norm(diff_dP) < 1E-8)
		self.assertTrue(LA.norm(diff_dP2) < 1E-8)

class TestFemInterval(unittest.TestCase):
	def test_1d_uniform_interval(self):
		from mozart.mesh.rectangle import interval
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

	def test_VandermondeM1D(self):
		from mozart.poisson.fem.interval import VandermondeM1D
		r = np.array([-1.0, 1.0])
		V1D = VandermondeM1D(1,r)
		diff_V1D = V1D - np.array([[0.707106781186548, -1.224744871391589], [0.707106781186548, 1.224744871391589]])
		self.assertTrue(LA.norm(diff_V1D) < 1E-8)

	def test_DVandermondeM1D(self):
		from mozart.poisson.fem.interval import DVandermondeM1D
		r = np.array([-1.0, 1.0])
		DVr = DVandermondeM1D(1,r)
		diff_DVr = DVr - np.array([[0.0, 1.224744871391589], [0.0, 1.224744871391589]])
		self.assertTrue(LA.norm(diff_DVr) < 1E-8)

	def test_Dmatrix1D(self):
		from mozart.poisson.fem.interval import Dmatrix1D
		from mozart.poisson.fem.interval import VandermondeM1D
		r = np.array([-1.0, 1.0])
		V = VandermondeM1D(1,r)
		Dr = Dmatrix1D(1,r,V)
		diff_Dr = Dr - np.array([[-0.5, 0.5], [-0.5, 0.5]])
		self.assertTrue(LA.norm(diff_Dr) < 1E-8)

	def test_solve(self):
		from mozart.mesh.rectangle import interval
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
		from mozart.mesh.rectangle import interval
		from mozart.poisson.fem.interval import solve
		from mozart.poisson.fem.interval import computeError
		N = 2
		iter = 4
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

class TestTecplot(unittest.TestCase):
	def test_tecplot_triangle(self):
		from mozart.common.etc import tecplot_triangle
		c4n = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
		n4e = np.array([[0, 1, 2]])
		u = np.array([10, 20, 30])
		tecplot_triangle('sample.dat', c4n, n4e, u)
		self.assertTrue(True)

class TestCommonModuleMethods(unittest.TestCase):
	def test_prefix_by_os(self):
		answer_sheet ={"linux" : "linux", "linux32" : "linux", 
			"darwin" : "osx", "win32" : "win64"}

		from mozart.common.etc import prefix_by_os
		for case, answer in answer_sheet.items():
			res = prefix_by_os(case)
			print(case, answer, res)
			self.assertEqual(res, answer)

	def test_benchmark01_sample(self):
		from mozart.poisson.fem.triangle import sample
		sample()
		self.assertTrue(True)

class TestFemRectangle(unittest.TestCase):
	def test_2d_uniform_rectangle(self):
		from mozart.mesh.rectangle import rectangle
		x1, x2, y1, y2, Mx, My, N = (0, 1, 0, 1, 2, 2, 1)
		c4n, ind4e, n4e, n4db = rectangle(x1,x2,y1,y2,Mx,My,N)
		diff_c4n = c4n - np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],[0.0,0.5],[0.5,0.5],[1.0,0.5],[0.0,1.0],[0.5,1.0],[1.0,1.0]])
		diff_ind4e = ind4e - np.array([[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]])
		diff_n4e = n4e - np.array([[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]])
		diff_n4db = n4db - np.array([0,1,2,3,5,6,7,8])
		
		self.assertTrue(LA.norm(diff_c4n) < 1E-8)
		self.assertTrue(LA.norm(diff_n4e) < 1E-8)
		self.assertTrue(LA.norm(diff_n4db) < 1E-8)
		self.assertTrue(LA.norm(diff_ind4e) < 1E-8)