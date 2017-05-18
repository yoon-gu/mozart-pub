import unittest
import numpy as np
from numpy import linalg as LA
import numpy.testing as npt

def test_nJacobiP():
	from mozart.poisson.fem.common import nJacobiP
	x = np.linspace(-1,1,5)
	P = nJacobiP(x,0,0,0)
	ref_P = 0.707106781186548 * np.ones(5,float)
	P2 = nJacobiP(x,0,0,1)
	ref_P2 = 1.224744871391589*x
	P3 = nJacobiP(x,0,0,2)
	ref_P3 = np.array([1.581138830084190, -0.197642353760524, -0.790569415042095, -0.197642353760524, 1.581138830084190])
	npt.assert_almost_equal(P, ref_P, decimal=8)
	npt.assert_almost_equal(P2, ref_P2, decimal=8)
	npt.assert_almost_equal(P3, ref_P3, decimal=8)

def test_DnJacobiP():
	from mozart.poisson.fem.common import DnJacobiP
	x = np.linspace(-1,1,5)
	dP = DnJacobiP(x,0,0,0)
	ref_dP = np.zeros(5,float)
	dP2 = DnJacobiP(x,0,0,1)
	ref_dP2 = 1.224744871391589*np.ones(5,float)
	npt.assert_almost_equal(dP, ref_dP, decimal=8)
	npt.assert_almost_equal(dP2, ref_dP2, decimal=8)

def test_nJacobiGQ():
	from mozart.poisson.fem.common import nJacobiGQ
	x0, w0 = nJacobiGQ(0,0,0)
	ref_x0 = 0.0
	ref_w0 = 2.0
	x1, w1 = nJacobiGQ(0,0,1)
	ref_x1 = np.array([-np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)])
	ref_w1 = np.array([1.0, 1.0])
	x2, w2 = nJacobiGQ(0,0,2)
	ref_x2 = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
	ref_w2 = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
	x3, w3 = nJacobiGQ(1,1,1)
	ref_x3 = np.array([-0.447213595499958,   0.447213595499958])
	ref_w3 = np.array([0.666666666666667,   0.666666666666667])
	npt.assert_almost_equal(x0, ref_x0, decimal=8)
	npt.assert_almost_equal(w0, ref_w0, decimal=8)
	npt.assert_almost_equal(x1, ref_x1, decimal=8)
	npt.assert_almost_equal(w1, ref_w1, decimal=8)
	npt.assert_almost_equal(x2, ref_x2, decimal=8)
	npt.assert_almost_equal(w2, ref_w2, decimal=8)
	npt.assert_almost_equal(x3, ref_x3, decimal=8)
	npt.assert_almost_equal(w3, ref_w3, decimal=8)

def test_nJacobiGL():
	from mozart.poisson.fem.common import nJacobiGL
	x0 = nJacobiGL(0,0,0)
	ref_x0 = 0.0
	x1 = nJacobiGL(0,0,1)
	ref_x1 = np.array([-1.0, 1.0])
	x2 = nJacobiGL(0,0,2)
	ref_x2 = np.array([-1.0, 0.0, 1.0])
	npt.assert_almost_equal(x0, ref_x0, decimal=8)
	npt.assert_almost_equal(x1, ref_x1, decimal=8)
	npt.assert_almost_equal(x2, ref_x2, decimal=8)

def test_VandermondeM1D():
	from mozart.poisson.fem.common import VandermondeM1D
	r = np.array([-1.0, 1.0])
	V1D = VandermondeM1D(1,r)
	ref_V1D = np.array([[0.707106781186548, -1.224744871391589], [0.707106781186548, 1.224744871391589]])
	npt.assert_almost_equal(V1D, ref_V1D, decimal=8)

def test_DVandermondeM1D():
	from mozart.poisson.fem.common import DVandermondeM1D
	r = np.array([-1.0, 1.0])
	DVr = DVandermondeM1D(1,r)
	ref_DVr = np.array([[0.0, 1.224744871391589], [0.0, 1.224744871391589]])
	npt.assert_almost_equal(DVr, ref_DVr, decimal=8)

def test_Dmatrix1D():
	from mozart.poisson.fem.common import Dmatrix1D, VandermondeM1D
	r = np.array([-1.0, 1.0])
	V = VandermondeM1D(1,r)
	Dr = Dmatrix1D(1,r,V)
	ref_Dr = np.array([[-0.5, 0.5], [-0.5, 0.5]])
	npt.assert_almost_equal(Dr, ref_Dr, decimal=8)

def test_RefNodes_Tri():
	from mozart.poisson.fem.common import RefNodes_Tri
	N = 0
	r0, s0 = RefNodes_Tri(N)
	ref_r0 = np.array([-1.0/3])
	ref_s0 = np.array([-1.0/3])
	N = 3
	r, s = RefNodes_Tri(N)
	ref_r = np.array([-1, -1.0/3, 1.0/3, 1, -1, -1.0/3, 1.0/3, -1, -1.0/3, -1])
	ref_s = np.array([-1, -1, -1, -1, -1.0/3, -1.0/3, -1.0/3, 1.0/3, 1.0/3, 1])
	npt.assert_almost_equal(r0, ref_r0, decimal=8)
	npt.assert_almost_equal(s0, ref_s0, decimal=8)
	npt.assert_almost_equal(r, ref_r, decimal=8)
	npt.assert_almost_equal(s, ref_s, decimal=8)

def test_rs2ab():
	from mozart.poisson.fem.common import RefNodes_Tri, rs2ab
	N = 3
	r, s = RefNodes_Tri(N)
	a, b = rs2ab(r,s)
	ref_a = np.array([-1, -1.0/3, 1.0/3, 1, -1, 0, 1, -1, 1, -1])
	ref_b = np.array([-1, -1, -1, -1, -1.0/3, -1.0/3, -1.0/3, 1.0/3, 1.0/3, 1])
	npt.assert_almost_equal(a, ref_a, decimal=8)
	npt.assert_almost_equal(b, ref_b, decimal=8)

def test_Simplex2DP():
	from mozart.poisson.fem.common import Simplex2DP
	a = np.array([0,1])
	b = np.array([2,3])
	p = Simplex2DP(a,b,0,0)
	ref_p = np.array([ 0.70710678, 0.70710678])
	npt.assert_almost_equal(p, ref_p, decimal=8)

def test_Vandermonde2D():
	from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D
	N = 2
	r, s = RefNodes_Tri(N)
	V2D = Vandermonde2D(N,r,s)
	ref_V2D = np.array([[ 0.707106781186548,  -1.              ,   1.224744871391590,  -1.732050807568878,   2.121320343559643,   2.738612787525831],
		[ 0.707106781186548,  -1.              ,   1.224744871391590,                   0,                   0,  -1.369306393762915],
		[ 0.707106781186548,  -1.              ,   1.224744871391590,   1.732050807568878,  -2.121320343559643,   2.738612787525831],
		[ 0.707106781186548,   0.500000000000000,  -0.612372435695795,  -0.866025403784439,  -1.590990257669732,   0.684653196881458],
		[ 0.707106781186548,   0.500000000000000,  -0.612372435695795,   0.866025403784439,   1.590990257669732,   0.684653196881458],
		[ 0.707106781186548,   2.000000000000001,   3.674234614174769,                   0,                   0,                   0]])
	npt.assert_almost_equal(V2D, ref_V2D, decimal=8)

def test_GradSimplex2DP():
	from mozart.poisson.fem.common import RefNodes_Tri, rs2ab, GradSimplex2DP
	N = 2
	r, s = RefNodes_Tri(N)
	a, b = rs2ab(r,s)
	dmodedr, dmodeds = GradSimplex2DP(a,b,1,1)
	ref_dmodedr = np.array([-2.121320343559642, -2.121320343559642, -2.121320343559642,
		 3.181980515339464,  3.181980515339464,  8.485281374238570])
	ref_dmodeds = np.array([-6.363961030678929, -1.060660171779821,  4.242640687119286,
		-1.060660171779822,  4.242640687119286,  4.242640687119285])
	npt.assert_almost_equal(dmodedr, ref_dmodedr, decimal=8)
	npt.assert_almost_equal(dmodeds, ref_dmodeds, decimal=8)

def test_GradVandermonde2D():
	from mozart.poisson.fem.common import RefNodes_Tri, GradVandermonde2D
	N = 2
	r, s = RefNodes_Tri(N)
	V2Dr, V2Ds = GradVandermonde2D(N,r,s)
	ref_V2Dr = np.array([[ 0.                 ,  0.                 ,  0.                 ,
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
	ref_V2Ds = np.array([[0.               ,  1.5              , -4.898979485566358,
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
	npt.assert_almost_equal(V2Dr, ref_V2Dr, decimal=8)
	npt.assert_almost_equal(V2Ds, ref_V2Ds, decimal=8)

def test_Dmatrices2D():
	from mozart.poisson.fem.common import RefNodes_Tri, Vandermonde2D, Dmatrices2D
	N = 2
	r, s = RefNodes_Tri(N)
	V = Vandermonde2D(N,r,s)
	Dr, Ds = Dmatrices2D(N,r,s,V)
	ref_Dr = np.array([[-1.5,  2., -0.5,  0.,  0.,  0.],
		[-0.5,  0.,  0.5,  0.,  0.,  0.],
		[ 0.5, -2.,  1.5,  0.,  0.,  0.],
		[-0.5,  1., -0.5, -1.,  1.,  0.],
		[ 0.5, -1.,  0.5, -1.,  1.,  0.],
		[ 0.5,  0., -0.5, -2.,  2.,  0.]])
	ref_Ds = np.array([[-1.5,  0.,  0.,  2.,  0.,  -0.5],
		[-0.5, -1.,  0.,  1.,  1., -0.5],
		[ 0.5, -2.,  0.,  0.,  2., -0.5],
		[-0.5,  0.,  0.,  0.,  0.,  0.5],
		[ 0.5, -1.,  0., -1.,  1.,  0.5],
		[ 0.5,  0.,  0., -2.,  0.,  1.5]])
	npt.assert_almost_equal(Dr, ref_Dr, decimal=8)
	npt.assert_almost_equal(Ds, ref_Ds, decimal=8)

def test_RefNodes_Rect():
	from mozart.poisson.fem.common import RefNodes_Rect
	r, s = RefNodes_Rect(3)
	ref_r = np.matrix('[-1;-0.447213595499958;0.447213595499958;1;-1;-0.447213595499958;0.447213595499958;1;-1;\
		-0.447213595499958;0.447213595499958;1;-1;-0.447213595499958;0.447213595499958;1]')
	ref_s = np.matrix('[-1;-1;-1;-1;-0.447213595499958;-0.447213595499958;-0.447213595499958;-0.447213595499958;\
		0.447213595499958;0.447213595499958;0.447213595499958;0.447213595499958;1;1;1;1]')
	npt.assert_almost_equal(r, ref_r.T, decimal=8)
	npt.assert_almost_equal(s, ref_s.T, decimal=8)

def test_Vandermonde2D_Rect():
	from mozart.poisson.fem.common import RefNodes_Rect, Vandermonde2D_Rect
	degree = 2
	r, s = RefNodes_Rect(degree)
	V2D = Vandermonde2D_Rect(degree, r, s)
	ref_V2D = np.matrix('[0.5 -0.866025403784439 1.1180339887499 -0.866025403784439 1.5 -1.93649167310371 1.1180339887499 -1.93649167310371 2.5;\
		0.5 -0.866025403784439 1.1180339887499 0 0 0 -0.559016994374947 0.968245836551854 -1.25;\
		0.5 -0.866025403784439 1.1180339887499 0.866025403784439 -1.5 1.93649167310371 1.1180339887499 -1.93649167310371 2.5;\
		0.5 0 -0.559016994374947 -0.866025403784439 0 0.968245836551854 1.1180339887499 0 -1.25;\
		0.5 0 -0.559016994374947 0 0 0 -0.559016994374947 0 0.625;\
		0.5 0 -0.559016994374947 0.866025403784439 0 -0.968245836551854 1.1180339887499 0 -1.25;\
		0.5 0.866025403784439 1.1180339887499 -0.866025403784439 -1.5 -1.93649167310371 1.1180339887499 1.93649167310371 2.5;\
		0.5 0.866025403784439 1.1180339887499 0 0 0 -0.559016994374947 -0.968245836551854 -1.25;\
		0.5 0.866025403784439 1.1180339887499 0.866025403784439 1.5 1.93649167310371 1.1180339887499 1.93649167310371 2.5]')
	
	npt.assert_almost_equal(V2D, ref_V2D, decimal=8)

def test_GradVandermonde2D_Rec():
	from mozart.poisson.fem.common import RefNodes_Rect, GradVandermonde2D_Rect
	degree = 2
	r,s = RefNodes_Rect(degree)
	V2Dr, V2Ds = GradVandermonde2D_Rect(degree, r, s)
	ref_V2Dr = np.matrix('[0 0 0 0.866025403784439 -1.5 1.93649167310371 -3.35410196624968 5.80947501931113 -7.5;\
		0 0 0 0.866025403784439 -1.5 1.93649167310371 0 0 0;\
		0 0 0 0.866025403784439 -1.5 1.93649167310371 3.35410196624968 -5.80947501931113 7.5;\
		0 0 0 0.866025403784439 0 -0.968245836551854 -3.35410196624968 0 3.75;\
		0 0 0 0.866025403784439 0 -0.968245836551854 0 0 0;0 0 0 0.866025403784439 0 -0.968245836551854 3.35410196624968 0 -3.75;\
		0 0 0 0.866025403784439 1.5 1.93649167310371 -3.35410196624968 -5.80947501931113 -7.5;\
		0 0 0 0.866025403784439 1.5 1.93649167310371 0 0 0;0 0 0 0.866025403784439 1.5 1.93649167310371 3.35410196624968 5.80947501931113 7.5]')
	ref_V2Ds = np.matrix('[0 0.866025403784439 -3.35410196624968 0 -1.5 5.80947501931113 0 1.93649167310371 -7.5;\
		0 0.866025403784439 -3.35410196624968 0 0 0 0 -0.968245836551854 3.75;\
	0 0.866025403784439 -3.35410196624968 0 1.5 -5.80947501931113 0 1.93649167310371 -7.5;\
	0 0.866025403784439 0 0 -1.5 0 0 1.93649167310371 0;\
	0 0.866025403784439 0 0 0 0 0 -0.968245836551854 0;\
	0 0.866025403784439 0 0 1.5 0 0 1.93649167310371 0;\
	0 0.866025403784439 3.35410196624968 0 -1.5 -5.80947501931113 0 1.93649167310371 7.5;\
	0 0.866025403784439 3.35410196624968 0 0 0 0 -0.968245836551854 -3.75;\
	0 0.866025403784439 3.35410196624968 0 1.5 5.80947501931113 0 1.93649167310371 7.5]')
	npt.assert_almost_equal(V2Dr, ref_V2Dr, decimal=8)
	npt.assert_almost_equal(V2Ds, ref_V2Ds, decimal=8)

def test_Dmatrices2D_Rect():
	from mozart.poisson.fem.common import RefNodes_Rect, Vandermonde2D_Rect, Dmatrices2D_Rect
	degree = 2
	r,s = RefNodes_Rect(degree)
	V = Vandermonde2D_Rect(degree, r, s)
	Dr, Ds = Dmatrices2D_Rect(degree, r, s, V)
	ref_Dr = np.matrix('[-1.5 2 -0.5 7.89491928622334e-17 -1.57898385724467e-16 7.89491928622333e-17 -6.04056724878456e-17 1.20811344975691e-16 -6.04056724878456e-17;\
		-0.5 0 0.5 0 0 0 0 0 0;\
		0.5 -2 1.5 -7.89491928622334e-17 1.57898385724467e-16 -7.89491928622333e-17 6.04056724878456e-17 -1.20811344975691e-16 6.04056724878456e-17;\
		-1.11022302462516e-16 -1.11022302462516e-16 2.77555756156289e-17 -1.5 2 -0.5 1.11022302462516e-16 -1.11022302462516e-16 2.77555756156289e-17;\
		0 -9.86864910777917e-18 2.77555756156289e-17 -0.5 1.97372982155583e-17 0.5 -2.77555756156289e-17 -9.86864910777917e-18 2.77555756156289e-17;\
		-2.22044604925031e-16 1.11022302462516e-16 1.38777878078145e-16 0.5 -2 1.5 -1.94289029309402e-16 1.11022302462516e-16 8.32667268468867e-17;\
		-1.66533453693773e-16 4.44089209850063e-16 -2.22044604925031e-16 7.89491928622334e-17 -1.57898385724467e-16 7.89491928622333e-17 -1.5 2 -0.5;\
		-5.55111512312578e-17 0 5.55111512312578e-17 0 0 0 -0.5 0 0.5;\
		-2.22044604925031e-16 -1.57898385724467e-16 1.66533453693773e-16 -7.89491928622334e-17 1.57898385724467e-16 -7.89491928622333e-17 0.5 -2 1.5]')
	ref_Ds = np.matrix('[-1.5 7.89491928622334e-17 -6.04056724878456e-17 2 -1.57898385724467e-16 1.20811344975691e-16 -0.5 7.89491928622333e-17 -6.04056724878456e-17;\
		0 -1.5 8.32667268468867e-17 -1.11022302462516e-16 2 -1.11022302462516e-16 2.77555756156289e-17 -0.5 2.77555756156289e-17;\
		-1.66533453693773e-16 7.89491928622334e-17 -1.5 4.44089209850063e-16 -1.57898385724467e-16 2 -2.77555756156289e-16 7.89491928622333e-17 -0.5;\
		-0.5 0 0 0 0 0 0.5 0 0;\
		-5.55111512312578e-17 -0.5 -2.28212510617393e-17 -9.86864910777917e-18 1.97372982155583e-17 -9.86864910777917e-18 3.26899001695185e-17 0.5 3.26899001695185e-17;\
		-5.55111512312578e-17 0 -0.5 0 0 0 5.55111512312578e-17 0 0.5;\
		0.5 -7.89491928622334e-17 6.04056724878456e-17 -2 1.57898385724467e-16 -1.20811344975691e-16 1.5 -7.89491928622333e-17 6.04056724878456e-17;\
		-2.22044604925031e-16 0.5 -1.38777878078145e-16 1.11022302462516e-16 -2 1.11022302462516e-16 8.32667268468867e-17 1.5 2.77555756156289e-17;\
		-2.22044604925031e-16 -7.89491928622334e-17 0.5 -2.22044604925031e-16 1.57898385724467e-16 -2 3.33066907387547e-16 -7.89491928622333e-17 1.5]')
	npt.assert_almost_equal(Dr, ref_Dr, decimal=8)
	npt.assert_almost_equal(Ds, ref_Ds, decimal=8)

def test_RefNodes_Cube():
	from mozart.poisson.fem.common import RefNodes_Cube
	r, s, t = RefNodes_Cube(3)
	ref_r = np.array([-1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0,  1/3.0, 1.0, -1.0, \
					  -1/3.0,  1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, \
					  1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, \
					  -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, \
					  1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0])
	ref_s = np.array([-1.0, -1.0, -1.0, -1.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, 1/3.0, 1/3.0, 1/3.0, 1/3.0, 1.0, \
					  1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, 1/3.0,  1/3.0, \
					  1/3.0,  1/3.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, \
					  1/3.0,  1/3.0,  1/3.0,  1/3.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1/3.0, -1/3.0, \
					  -1/3.0, -1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0, 1.0, 1.0, 1.0, 1.0])
	ref_t = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,\
					  -1.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, \
					  -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  \
					  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,  1/3.0,\
					  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1])
	npt.assert_almost_equal(r, ref_r, decimal=8)
	npt.assert_almost_equal(s, ref_s, decimal=8)
	npt.assert_almost_equal(t, ref_t, decimal=8)

def test_RefNodes_Rec():
	from mozart.poisson.fem.common import RefNodes_Rec
	r, s = RefNodes_Rec(3)
	ref_r = np.array([-1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0, 1/3.0, 1.0, -1.0, -1/3.0,  1/3.0, 1.0, -1.0, \
					  -1/3.0,  1/3.0, 1.0])
	ref_s = np.array([-1.0, -1.0, -1.0, -1.0, -1/3.0, -1/3.0, -1/3.0, -1/3.0, 1/3.0, 1/3.0, 1/3.0, 1/3.0, 1.0, \
					  1.0, 1.0, 1.0])
	npt.assert_almost_equal(r, ref_r, decimal=8)
	npt.assert_almost_equal(s, ref_s, decimal=8)

def test_Simplex3DP_Cube():
	from mozart.poisson.fem.common import Simplex3DP_Cube
	a = np.array([0,1])
	b = np.array([2,3])
	c = np.array([3,4])
	p = Simplex3DP_Cube(a,b,c,0,0,0)
	ref_p = np.array([(np.sqrt(2)/2)**3, (np.sqrt(2)/2)**3])
	npt.assert_almost_equal(p, ref_p, decimal=8)

def test_GradSimplex3DP_Cube():
	from mozart.poisson.fem.common import RefNodes_Cube, GradSimplex3DP_Cube
	N = 2
	r, s, t = RefNodes_Cube(N)
	dmodedr, dmodeds, dmodedt = GradSimplex3DP_Cube(r,s,t,1,1,1)
	ref_dmodedr = np.array([1.83711730708738, 1.83711730708738, 1.83711730708738, 0., 0., 0., 
		-1.83711730708738, -1.83711730708738, -1.83711730708738, 0., 0., 0., 0., 0., 0., 0., 
		0., 0., -1.83711730708738, -1.83711730708738, -1.83711730708738, 0., 0., 0., 
		1.83711730708738, 1.83711730708738, 1.83711730708738])
	ref_dmodeds = np.array([1.83711730708738, 0., -1.83711730708738, 1.83711730708738, 0., 
		-1.83711730708738, 1.83711730708738, 0., -1.83711730708738, 0., 0., 0., 0., 0., 0., 
		0., 0., 0., -1.83711730708738, 0., 1.83711730708738, -1.83711730708738, 0., 
		1.83711730708738, -1.83711730708738, 0., 1.83711730708738])
	ref_dmodedt = np.array([1.83711730708738, 0., -1.83711730708738, 0., 0., 0., -1.83711730708738, 
		0., 1.83711730708738, 1.83711730708738, 0., -1.83711730708738, 0., 0., 0., -1.83711730708738, 
		0., 1.83711730708738, 1.83711730708738, 0., -1.83711730708738, 0., 0., 0., -1.83711730708738, 
		0., 1.83711730708738])
	npt.assert_almost_equal(dmodedr, ref_dmodedr, decimal=8)
	npt.assert_almost_equal(dmodeds, ref_dmodeds, decimal=8)
	npt.assert_almost_equal(dmodedt, ref_dmodedt, decimal=8)