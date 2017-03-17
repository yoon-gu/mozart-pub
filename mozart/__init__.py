"""
Mozart
===============

``mozart`` is a software package written in ``Python`` for PDE Solver.

Example
-------
Examples can be given using either the ``Example`` or ``Examples``
sections. Sections support any reStructuredText formatting, including
literal blocks.

	$ python example_numpy.py


Section breaks are created with two blank lines. Section breaks are also
implicitly created anytime a new section starts. Section bodies *may* be
indented.

Notes
-----
	This is an example of an indented section. It's like any other section,
	but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created simply by
resuming unindented text.

"""
__author__ = ("Yoon-gu Hwang <yz0624@gmail.com>", 
	"Dong-Wook Shin <dwshin.yonsei@gmail.com>", 
	"Ji-Yeon Suh <suh91919@gmail.com>")

__all__ = ["greeting", "sample"]
from .Poisson import *