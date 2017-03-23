[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yoon-gu/mozart/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/mozart/badge/?version=latest)](http://mozart.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/yoon-gu/Mozart.svg?branch=master)](https://travis-ci.org/yoon-gu/Mozart)
[![Build status](https://ci.appveyor.com/api/projects/status/55et5ffdm46eyi2y?svg=true)](https://ci.appveyor.com/project/yoon-gu/mozart)
[![codecov](https://codecov.io/gh/yoon-gu/Mozart/branch/master/graph/badge.svg)](https://codecov.io/gh/yoon-gu/Mozart)
[![Coverage Status](https://coveralls.io/repos/github/yoon-gu/Mozart/badge.svg?branch=master)](https://coveralls.io/github/yoon-gu/Mozart?branch=master)

# Mozart
## Intallation
- run `pip install git+https://github.com/yoon-gu/Mozart.git`
- document : http://mozart.readthedocs.io/en/latest/

## Contribution
1. Fork
1. Write code
1. Add `docstring` for documentation. It should contain the following four things:
	1. Description of a function
	1. `Parameters`
	1. `Returns` 
	1. `Example`
	1. For example
		```python
		def one_dim(c4n, n4e, n4Db, f, u_D, degree = 1):
		"""
		Computes the coordinates of nodes and elements.
		
		Parameters
			- ``c4n`` (``float64 array``) : coordinates
			- ``n4e`` (``int32 array``) : nodes for elements
			- ``n4Db`` (``int32 array``) : Dirichlet boundary nodes
			- ``f`` (``lambda``) : source term 
			- ``u_D`` (``lambda``) : Dirichlet boundary condition
			- ``degree`` (``int32``) : Polynomial degree

		Returns
			- ``x`` (``float64 array``) : solution

		Example
			>>> N = 3
			>>> c4n, n4e = unit_interval(N)
			>>> n4Db = [0, N-1]
			>>> f = lambda x: np.ones_like(x)
			>>> u_D = lambda x: np.zeros_like(x)
			>>> from mozart.poisson.solve import one_dim
			>>> x = one_dim(c4n, n4e, n4Db, f, u_D)
			>>> print(x)
			array([ 0.   ,  0.125,  0.   ])
		"""
		```
1. Add test code for your code.
	- Without this, you will have failure for automation test system(`cdoecov`).
	- run `cd /path/to/Mozart/`
	- run `nosetests`
1. Check documentation by running 
	1. `cd /path/to/Mozart/`
	1. `python setup.py install`
	1. `cd /path/to/Mozart/docs`
	1. `make html`(UNIX) or `./make.bat html`(Windows)
	1. Open `Mozart/docs/_build/index.html`
