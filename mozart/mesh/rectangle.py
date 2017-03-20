import numpy as np
def unit_interval(N):
	"""
	Computes the coordinates of nodes and elements.
	
	Parameters
		- ``N`` (``int``) : Number of nodes

	Returns
		- ``c4n`` (``float array``) : coordinates of nodes
		- ``n4e`` (``int array``) : elements

	Example
		>>> c4n, n4e = unit_interval(4)
		>>> c4n 
		array([ 0.        ,  0.33333333,  0.66666667,  1.        ])
		>>> n4e
		array([[0, 1],
		   [1, 2],
		   [2, 3]])
	"""
	c4n = np.linspace(0, 1, N)
	n4e = np.array([[item,item+1] for item in range(0,N-1)], dtype=np.int32)
	return (c4n, n4e)

def unit_square(h):
	print("unit_square is called.")