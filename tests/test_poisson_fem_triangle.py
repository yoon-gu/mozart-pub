import numpy as np
import numpy.testing as npt

def test_getMatrix2D():
	from mozart.poisson.fem.triangle import getMatrix
	N = 1
	M_R, Srr_R, Srs_R, Ssr_R, Sss_R, Dr_R, Ds_R, M1D_R = getMatrix(N)
	ref_M = np.array([[1.0/3, 1.0/6, 1.0/6,], [1.0/6, 1.0/3, 1.0/6], [1.0/6, 1.0/6, 1.0/3]], dtype = np.float64)
	ref_Srr = np.array([[1.0/2, -1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0], [0.0, 0.0, 0.0]], dtype = np.float64)
	ref_Srs = np.array([[1.0/2, 0.0, -1.0/2], [-1.0/2, 0.0, 1.0/2], [0.0, 0.0, 0.0]], dtype = np.float64)
	ref_Ssr = np.array([[1.0/2, -1.0/2, 0.0], [0.0, 0.0, 0.0], [-1.0/2, 1.0/2, 0.0]], dtype = np.float64)
	ref_Sss = np.array([[1.0/2, 0.0, -1.0/2], [0.0, 0.0, 0.0], [-1.0/2, 0.0, 1.0/2]], dtype = np.float64)
	ref_Dr = np.array([[-1.0/2, 1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0], [-1.0/2, 1.0/2, 0.0]], dtype = np.float64)
	ref_Ds = np.array([[-1.0/2, 0.0, 1.0/2], [-1.0/2, 0.0, 1.0/2], [-1.0/2, 0.0, 1.0/2]], dtype = np.float64)
	ref_M1D = np.array([[2.0/3, 1.0/3], [1.0/3, 2.0/3]], dtype = np.float64)
	npt.assert_almost_equal(M_R, ref_M, decimal=8)
	npt.assert_almost_equal(Srr_R, ref_Srr, decimal=8)
	npt.assert_almost_equal(Srs_R, ref_Srs, decimal=8)
	npt.assert_almost_equal(Ssr_R, ref_Ssr, decimal=8)
	npt.assert_almost_equal(Sss_R, ref_Sss, decimal=8)
	npt.assert_almost_equal(Dr_R, ref_Dr, decimal=8)
	npt.assert_almost_equal(Ds_R, ref_Ds, decimal=8)
	npt.assert_almost_equal(M1D_R, ref_M1D, decimal=8)

def test_compute_n4s():
	from mozart.poisson.fem.triangle import compute_n4s
	n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
	n4s = compute_n4s(n4e)
	ref_n4s = np.array([[1, 3], [2, 4], [4, 6], [5, 7], [3, 0], [1, 4], [2, 5],
		[6, 3], [4, 7], [5, 8], [0, 1], [4, 3], [1, 2], [5, 4], [7, 6], [8, 7]], dtype = int)
	npt.assert_almost_equal(n4s, ref_n4s, decimal=8)

def test_compute_s4e():
	from mozart.poisson.fem.triangle import compute_s4e
	n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
	s4e = compute_s4e(n4e)
	ref_s4e = np.array([[0, 4, 10], [0, 5, 11], [1, 5, 12], [1, 6, 13],
		[2, 7, 11], [2, 8, 14], [3, 8, 13], [3, 9, 15]], dtype = int)
	npt.assert_almost_equal(s4e, ref_s4e, decimal=8)

def test_compute_e4s():
	from mozart.poisson.fem.triangle import compute_e4s
	n4e = np.array([[1, 3, 0], [3, 1, 4], [2, 4, 1], [4, 2, 5], [4, 6, 3], [6, 4, 7], [5, 7, 4], [7, 5, 8]])
	e4s = compute_e4s(n4e)
	ref_e4s = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, -1], [1, 2],
		[3, -1], [4, -1], [5, 6], [7, -1], [0, -1], [1, 4], [2, -1], [3, 6], [5, -1], [7, -1]], dtype = int)
	npt.assert_almost_equal(e4s, ref_e4s, decimal=8)

def test_refineUniformRed():
	from mozart.poisson.fem.triangle import refineUniformRed
	c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5]])
	n4e = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
	n4Db = np.array([[0, 1], [1, 2]])
	n4Nb = np.array([[2, 3], [3, 0]])
	c4n, n4e, n4Db, n4Nb = refineUniformRed(c4n, n4e, n4Db, n4Nb)
	ref_c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0.5, 0.5], [0.5, 0.],
		[1., 0.5], [0.5, 1.], [0., 0.5], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]], dtype = np.float64)
	ref_n4e = np.array([[0, 5, 12], [5, 1, 9], [9, 12, 5], [12, 9, 4], [1, 6, 9], [6, 2, 10], [10, 9, 6],
		[9, 10, 4], [2, 7, 10], [7, 3, 11], [11, 10, 7], [10, 11, 4], [3, 8, 11], [8, 0, 12], [12, 11, 8], [11, 12, 4]], dtype = int)
	ref_n4Db = np.array([[0, 5], [5, 1], [1, 6], [6, 2]], dtype = int)
	ref_n4Nb = np.array([[2, 7], [7, 3], [3, 8], [8, 0]], dtype = int)
	npt.assert_almost_equal(c4n, ref_c4n, decimal=8)
	npt.assert_almost_equal(n4e, ref_n4e, decimal=8)
	npt.assert_almost_equal(n4Db, ref_n4Db, decimal=8)
	npt.assert_almost_equal(n4Nb, ref_n4Nb, decimal=8)

def test_getIndex():
	from mozart.poisson.fem.triangle import getIndex
	N = 3
	c4n = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
	n4e = np.array([[1, 3, 0], [3, 1, 2]])
	n4sDb = np.array([[0, 1], [2, 3], [3, 4]])
	n4sNb = np.array([[1, 2]])
	c4nNew, ind4e, ind4Db, ind4Nb = getIndex(N, c4n, n4e, n4sDb, n4sNb)
	ref_c4nNew = np.array([[0., 0.], [1.,  0.], [1., 1.], [ 0., 1.], [2.0/3, 1.0/3],
		[1.0/3, 2.0/3], [0., 2.0/3], [0., 1.0/3], [1., 1.0/3], [1., 2.0/3], [1.0/3, 0.],
		[2.0/3, 0.], [2.0/3, 1.], [1.0/3, 1.], [1.0/3, 1.0/3], [2.0/3, 2.0/3]], dtype = np.float64)
	ref_ind4e = np.array([[ 0, 10, 11,  1,  7, 14,  4,  6,  5,  3],
		[ 2, 12, 13,  3,  9, 15,  5,  8,  4,  1]], dtype = int)
	ref_ind4Db = np.array([ 0,  1,  2,  3,  6,  7, 10, 11, 12, 13], dtype = int)
	ref_ind4Nb = np.array([[1, 8, 9, 2]], dtype = int)
	npt.assert_almost_equal(c4nNew, ref_c4nNew, decimal=8)
	npt.assert_almost_equal(ind4e, ref_ind4e, decimal=8)
	npt.assert_almost_equal(ind4Db, ref_ind4Db, decimal=8)
	npt.assert_almost_equal(ind4Nb, ref_ind4Nb, decimal=8)
