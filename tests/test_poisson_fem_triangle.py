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