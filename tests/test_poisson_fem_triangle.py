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