import os
from os import path
import numpy as np

def prefix_by_os(platform="linux"):
	# OS Detection Code
	prefix = "linux"
	if platform == "linux" or platform == "linux32":
		prefix = "linux"
	elif platform == "darwin":
		prefix = "osx"
	elif platform == "win32":
		prefix = "win64"

	return prefix

def tecplot_triangle(filename, c4n, n4e, u):
	coord_x = c4n[:, 0]
	coord_y = c4n[:, 1]
	nrNodes, nrElems = c4n.shape[0], n4e.shape[0]

	header_str = """TITLE = "2D Finite Element Triangulation Plot"
VARIABLES = "X", "Y", "U"
ZONE T="P_1", DATAPACKING=POINT, NODES={0}, ELEMENTS={1}, ZONETYPE=FETRIANGLE
""".format(nrNodes, nrElems)

	data_str = ""
	for k in range(0, nrNodes):
		data_str += "{0} {1} {2}\n".format(coord_x[k], coord_y[k], u[k])

	np.savetxt(filename, (n4e+1).reshape((nrElems, 3)), 
		fmt='%d',
		header=header_str + data_str, comments="")