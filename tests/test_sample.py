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
		authors = ('Yoon-gu Hwang <yz0624@gmail.com>', 'Dong-Wook Shin <dwshin.yonsei@gmail.com>')
		self.assertEqual(mz.__author__, authors)

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
