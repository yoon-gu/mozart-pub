import unittest

class TestStocMethods(unittest.TestCase):

	def test_test(self):
		self.assertTrue(True)

	def test_import(self):
		import mozart as mz
		self.assertTrue(True)

	def test_authors(self):
		import mozart as mz
		authors = ('Yoon-gu Hwang <yz0624@gmail.com>', 'Dong-Wook Shin <dwshin.yonsei@gmail.com>', 'Ji-Yeon Suh <suh91919@gmail.com>')
		self.assertEqual(mz.__author__, authors)
	def test_poisson_square_2d(self):
		from mozart.mesh.rectangle import unit_square
		unit_square(0.1)
		# from mozart.poisson.solve import one_dim
		# from mozart.poisson.solve import two_dim
		# from mozart.poisson.solve import three_dim
		# one_dim(None, None, None, None)
		# two_dim(None, None, None, None)
		# three_dim(None, None, None, None)
		self.assertTrue(True)