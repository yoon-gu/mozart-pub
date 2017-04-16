import unittest
import numpy as np
from numpy import linalg as LA
import numpy.testing as npt

class TestFemMeshInterval(unittest.TestCase):
	def test_1d_uniform_interval(self):
		from mozart.mesh.interval import interval
		a, b, M, N = (0, 1, 4, 2)
		c4n, n4e, n4db, ind4e = interval(a,b,M,N)
		npt.assert_almost_equal(c4n, np.linspace(a,b,M*N+1), decimal = 8)
		npt.assert_almost_equal(n4e, np.array([[0,2], [2,4], [4,6], [6,8]]), decimal = 8)
		npt.assert_almost_equal(n4db, np.array([0, 8]), decimal = 8)
		npt.assert_almost_equal(ind4e, np.array([[0,1,2], [2,3,4], [4,5,6], [6,7,8]]), decimal = 8)