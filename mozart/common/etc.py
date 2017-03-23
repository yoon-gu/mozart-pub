from os import path
from ctypes import CDLL, c_double, c_void_p, c_int, c_bool
import mozart as mz

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