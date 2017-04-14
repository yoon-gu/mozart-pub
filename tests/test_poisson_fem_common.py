import unittest
import numpy as np
from numpy import linalg as LA

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
		self.assertAlmostEqual(LA.norm(diff_P), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_P2), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_P3), 0.0, 8)

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
		self.assertAlmostEqual(LA.norm(diff_x0), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_w0), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_x1), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_w1), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_x2), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_w2), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_x3), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_w3), 0.0, 8)

	def test_nJacobiGL(self):
		from mozart.poisson.fem.common import nJacobiGL
		x0 = nJacobiGL(0,0,0)
		diff_x0 = x0 - 0.0;
		x1 = nJacobiGL(0,0,1)
		diff_x1 = x1 - np.array([-1.0, 1.0])
		x2 = nJacobiGL(0,0,2)
		diff_x2 = x2 - np.array([-1.0, 0.0, 1.0])
		self.assertAlmostEqual(LA.norm(diff_x0), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_x1), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_x2), 0.0, 8)

	def test_DnJacobiP(self):
		from mozart.poisson.fem.common import DnJacobiP
		x = np.linspace(-1,1,5)
		dP = DnJacobiP(x,0,0,0)
		diff_dP = dP - np.zeros(5,float)
		dP2 = DnJacobiP(x,0,0,1)
		diff_dP2 = dP2 - 1.224744871391589*np.ones(5,float)
		self.assertAlmostEqual(LA.norm(diff_dP), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_dP2), 0.0, 8)

	def test_VandermondeM1D(self):
		from mozart.poisson.fem.common import VandermondeM1D
		r = np.array([-1.0, 1.0])
		V1D = VandermondeM1D(1,r)
		diff_V1D = V1D - np.array([[0.707106781186548, -1.224744871391589], [0.707106781186548, 1.224744871391589]])
		self.assertAlmostEqual(LA.norm(diff_V1D), 0.0, 8)

	def test_DVandermondeM1D(self):
		from mozart.poisson.fem.common import DVandermondeM1D
		r = np.array([-1.0, 1.0])
		DVr = DVandermondeM1D(1,r)
		diff_DVr = DVr - np.array([[0.0, 1.224744871391589], [0.0, 1.224744871391589]])
		self.assertAlmostEqual(LA.norm(diff_DVr), 0.0, 8)

	def test_Dmatrix1D(self):
		from mozart.poisson.fem.common import Dmatrix1D, VandermondeM1D
		r = np.array([-1.0, 1.0])
		V = VandermondeM1D(1,r)
		Dr = Dmatrix1D(1,r,V)
		diff_Dr = Dr - np.array([[-0.5, 0.5], [-0.5, 0.5]])
		self.assertAlmostEqual(LA.norm(diff_Dr), 0.0, 8)

	def test_RefNodes_Tri(self):
		from mozart.poisson.fem.common import RefNodes_Tri
		N = 0
		r0, s0 = RefNodes_Tri(N)
		diff_r0 = r0 - np.array([-1.0/3])
		diff_s0 = s0 - np.array([-1.0/3])
		N = 3
		r, s = RefNodes_Tri(N)
		diff_r = r - np.array([-1, -1.0/3, 1.0/3, 1, -1, -1.0/3, 1.0/3, -1, -1.0/3, -1])
		diff_s = s - np.array([-1, -1, -1, -1, -1.0/3, -1.0/3, -1.0/3, 1.0/3, 1.0/3, 1])
		self.assertAlmostEqual(LA.norm(diff_r), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_s), 0.0, 8)

	def test_rs2ab(self):
		from mozart.poisson.fem.common import RefNodes_Tri, rs2ab
		N = 3
		r, s = RefNodes_Tri(N)
		a, b = rs2ab(r,s)
		diff_a = a - np.array([-1, -1.0/3, 1.0/3, 1, -1, 0, 1, -1, 1, -1])
		diff_b = b - np.array([-1, -1, -1, -1, -1.0/3, -1.0/3, -1.0/3, 1.0/3, 1.0/3, 1])
		self.assertAlmostEqual(LA.norm(diff_a), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_b), 0.0, 8)

	def test_Simplex2DP(self):
		from mozart.poisson.fem.common import Simplex2DP
		a = np.array([0,1])
		b = np.array([2,3])
		p = Simplex2DP(a,b,0,0)
		diff_p = p - np.array([ 0.70710678, 0.70710678])
		self.assertAlmostEqual(LA.norm(diff_p), 0.0, 8)

	def test_Vandermonde2D(self):
		from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D
		N = 2
		r, s = RefNodes_Tri(N)
		V2D = Vandermonde2D(N,r,s)
		diff_V2D = V2D - np.array([[ 0.707106781186548,  -1.              ,   1.224744871391590,  -1.732050807568878,   2.121320343559643,   2.738612787525831],
			[ 0.707106781186548,  -1.              ,   1.224744871391590,                   0,                   0,  -1.369306393762915],
			[ 0.707106781186548,  -1.              ,   1.224744871391590,   1.732050807568878,  -2.121320343559643,   2.738612787525831],
			[ 0.707106781186548,   0.500000000000000,  -0.612372435695795,  -0.866025403784439,  -1.590990257669732,   0.684653196881458],
			[ 0.707106781186548,   0.500000000000000,  -0.612372435695795,   0.866025403784439,   1.590990257669732,   0.684653196881458],
			[ 0.707106781186548,   2.000000000000001,   3.674234614174769,                   0,                   0,                   0]])
		self.assertAlmostEqual(LA.norm(diff_V2D), 0.0, 8)

	def test_GradSimplex2DP(self):
		from mozart.poisson.fem.common import RefNodes_Tri, rs2ab, GradSimplex2DP
		N = 2
		r, s = RefNodes_Tri(N)
		a, b = rs2ab(r,s)
		dmodedr, dmodeds = GradSimplex2DP(a,b,1,1)
		diff_dmodedr = dmodedr - np.array([-2.121320343559642, -2.121320343559642, -2.121320343559642,
			 3.181980515339464,  3.181980515339464,  8.485281374238570])
		diff_dmodeds = dmodeds - np.array([-6.363961030678929, -1.060660171779821,  4.242640687119286,
			-1.060660171779822,  4.242640687119286,  4.242640687119285])
		self.assertAlmostEqual(LA.norm(diff_dmodedr), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_dmodeds), 0.0, 8)

	def test_GradVandermonde2D(self):
		from mozart.poisson.fem.common import RefNodes_Tri, GradVandermonde2D
		N = 2
		r, s = RefNodes_Tri(N)
		V2Dr, V2Ds = GradVandermonde2D(N,r,s)
		diff_V2Dr = V2Dr - np.array([[ 0.                 ,  0.                 ,  0.                 ,
		      1.732050807568877, -2.121320343559642, -8.215838362577491],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877, -2.121320343559642,  0.               ],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877, -2.121320343559642,  8.215838362577491],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877,  3.181980515339464, -4.107919181288746],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877,  3.181980515339464,  4.107919181288746],
		    [ 0.               ,  0.               ,  0.               ,
		      1.732050807568877,  8.485281374238570,  0.               ]])
		diff_V2Ds = V2Ds - np.array([[0.               ,  1.5              , -4.898979485566358,
			  0.866025403784439, -6.363961030678929, -2.738612787525831],
			[ 0.               ,  1.5              , -4.898979485566358,
			  0.866025403784439, -1.060660171779821,  1.369306393762915],
			[ 0.               ,  1.500000000000000, -4.898979485566358,
			  0.866025403784439,  4.242640687119286,  5.477225575051659],
			[ 0.               ,  1.500000000000000,  1.224744871391589,
			  0.866025403784439, -1.060660171779822, -1.369306393762915],
			[ 0.               ,  1.500000000000000,  1.224744871391589,
			  0.866025403784439,  4.242640687119286,  2.738612787525830],
			[ 0.               ,  1.500000000000000,  7.348469228349536,
			  0.866025403784439,  4.242640687119285,  0.               ]])
		self.assertAlmostEqual(LA.norm(diff_V2Dr), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_V2Ds), 0.0, 8)

	def test_Dmatrices2D(self):
		from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D, Dmatrices2D
		N = 2
		r, s = RefNodes_Tri(N)
		V = Vandermonde2D(N,r,s)
		Dr, Ds = Dmatrices2D(N,r,s,V)
		diff_Dr = Dr - np.array([[-1.5,  2., -0.5,  0.,  0.,  0.],
		    [-0.5,  0.,  0.5,  0.,  0.,  0.],
		    [ 0.5, -2.,  1.5,  0.,  0.,  0.],
		    [-0.5,  1., -0.5, -1.,  1.,  0.],
		    [ 0.5, -1.,  0.5, -1.,  1.,  0.],
		    [ 0.5,  0., -0.5, -2.,  2.,  0.]])
		diff_Ds = Ds - np.array([[-1.5,  0.,  0.,  2.,  0.,  -0.5],
			[-0.5, -1.,  0.,  1.,  1., -0.5],
			[ 0.5, -2.,  0.,  0.,  2., -0.5],
			[-0.5,  0.,  0.,  0.,  0.,  0.5],
			[ 0.5, -1.,  0., -1.,  1.,  0.5],
			[ 0.5,  0.,  0., -2.,  0.,  1.5]])
		self.assertAlmostEqual(LA.norm(diff_Dr), 0.0, 8)
		self.assertAlmostEqual(LA.norm(diff_Ds), 0.0, 8)
