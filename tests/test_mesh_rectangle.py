import unittest
import numpy as np
import numpy.testing as npt

def test_2d_uniform_rectangle_degree_one():
	from mozart.mesh.rectangle import rectangle
	x1, x2, y1, y2, Mx, My, N = (0, 1, 0, 1, 2, 2, 1)
	c4n, ind4e, n4e, n4Db = rectangle(x1,x2,y1,y2,Mx,My,N)
	npt.assert_almost_equal(c4n, np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0],[0.0,0.5],[0.5,0.5],[1.0,0.5],[0.0,1.0],[0.5,1.0],[1.0,1.0]]), decimal=8)
	npt.assert_almost_equal(ind4e, np.array([[0,1,3,4],[1,2,4,5],[3,4,6,7],[4,5,7,8]]), decimal=8)
	npt.assert_almost_equal(n4e, np.array([[0,1,4,3],[1,2,5,4],[3,4,7,6],[4,5,8,7]]), decimal=8)
	npt.assert_almost_equal(n4Db, np.array([0,1,2,3,5,6,7,8]), decimal=8)

def test_2d_uniform_rectangle_degree_two():
	from mozart.mesh.rectangle import rectangle
	x1, x2, y1, y2, Mx, My, N = (0, 1, 0, 1, 2, 2, 2)
	c4n, ind4e, n4e, n4Db = rectangle(x1,x2,y1,y2,Mx,My,N)
	ref_c4n = np.matrix('[0 0;0.25 0;0.5 0;0.75 0;1 0;0 0.25;0.25 0.25;\
						0.5 0.25;0.75 0.25;1 0.25;0 0.5;0.25 0.5;0.5 0.5;\
						0.75 0.5;1 0.5;0 0.75;0.25 0.75;0.5 0.75;0.75 0.75;\
						1 0.75;0 1;0.25 1;0.5 1;0.75 1;1 1]')
	ref_ind4e = np.matrix('[0 1 2 5 6 7 10 11 12;2 3 4 7 8 9 12 13 14;10 11 12 15 16 17 20 21 22;12 13 14 17 18 19 22 23 24]')
	ref_n4e = np.matrix('[0 2 12 10;2 4 14 12;10 12 22 20;12 14 24 22]')
	ref_n4Db = np.squeeze(np.asarray(np.matrix('[0 1 2 3 4 5 9 10 14 15 19 20 21 22 23 24]')))
	npt.assert_almost_equal(c4n, ref_c4n, decimal=8)
	npt.assert_almost_equal(ind4e, ref_ind4e, decimal=8)
	npt.assert_almost_equal(n4e, ref_n4e, decimal=8)
	npt.assert_almost_equal(n4Db, ref_n4Db, decimal=8)