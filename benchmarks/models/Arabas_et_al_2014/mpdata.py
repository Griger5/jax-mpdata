# implementation based on Arabas et al. 2014 https://doi.org/10.3233/SPR-140379

real_t = "float32"

import numpy, numpy as np
from matplotlib import pyplot

class shift():
	def __init__(self, plus, mnus):
		self.plus = plus
		self.mnus = mnus
	def __radd__(self, arg): 
		return type(arg)(
			arg.start + self.plus, 
			arg.stop  + self.plus
		)
	def __rsub__(self, arg): 
		return type(arg)(
			arg.start - self.mnus, 
			arg.stop  - self.mnus
		)

one = shift(1,1)
hlf = shift(0,1)

def pi(d, *idx): 
	return (idx[d], idx[d-1])

def f(psi_l, psi_r, C):
	return (
		(C + abs(C)) * psi_l + 
		(C - abs(C)) * psi_r
	) / 2

def donorcell(d, psi, C, i, j):
	return (
		f(
		psi[pi(d, i,     j)], 
		psi[pi(d, i+one, j)], 
			C[pi(d, i+hlf, j)]
		) - 
		f(
		psi[pi(d, i-one, j)], 
		psi[pi(d, i,     j)], 
			C[pi(d, i-hlf, j)]
		) 
	)

def donorcell_op(psi, n, C, i, j):
	psi[n+1][i,j] = psi[n][i,j] - (
		donorcell(0, psi[n], C[0], i, j) +
		donorcell(1, psi[n], C[1], j, i)
	)

class Solver(object):
	# ctor-like method
	def __init__(self, bcx, bcy, nx, ny, hlo):
		self.n = 0
		self.hlo = hlo
		self.i = slice(hlo, nx + hlo)
		self.j = slice(hlo, ny + hlo)

		self.bcx = bcx(0, self.i, hlo)
		self.bcy = bcy(1, self.j, hlo)

		self.psi = (
			numpy.empty((
				ext(self.i, self.hlo).stop, 
				ext(self.j, self.hlo).stop
			), real_t),
			numpy.empty((
				ext(self.i, self.hlo).stop, 
				ext(self.j, self.hlo).stop
			), real_t)
		)

		self.C = (
			numpy.empty((
				ext(self.i, hlf).stop, 
				ext(self.j, self.hlo).stop
			), real_t),
			numpy.empty((
				ext(self.i, self.hlo).stop, 
				ext(self.j, hlf).stop
			), real_t)
		)

	# accessor methods
	def state(self):
		return self.psi[self.n][self.i, self.j]

	# helper methods invoked by solve()
	def courant(self,d):
		return self.C[d][:]

	def cycle(self):
		self.n  = (self.n + 1) % 2 - 2

	# integration logic
	def solve(self, nt):
		for t in range(nt):
			self.bcx.fill_halos(
				self.psi[self.n], ext(self.j, self.hlo)
			)
			self.bcy.fill_halos(
				self.psi[self.n], ext(self.i, self.hlo)
			)
			self.advop() 
			self.cycle()

#listing16
class cyclic(object):
# ctor
	def __init__(self, d, i, hlo): 
		self.d = d
		self.left_halo = slice(i.start-hlo, i.start    )
		self.rght_edge = slice(i.stop -hlo, i.stop     )
		self.rght_halo = slice(i.stop,      i.stop +hlo)
		self.left_edge = slice(i.start,     i.start+hlo)

# method invoked by the solver
	def fill_halos(self, psi, j):
		psi[pi(self.d, self.left_halo, j)] = (
		psi[pi(self.d, self.rght_edge, j)]
		)
		psi[pi(self.d, self.rght_halo, j)] = (
		psi[pi(self.d, self.left_edge, j)]
		)

#listing17
class solver_donorcell(Solver):
	def __init__(self, bcx, bcy, nx, ny):
		Solver.__init__(self, bcx, bcy, nx, ny, 1)

	def advop(self):
		donorcell_op(
			self.psi, self.n, 
			self.C, self.i, self.j
		)

def ext(r, n):
	if (type(n) == int) & (n == 1): 
		n = one
	return slice(
		(r - n).start, 
		(r + n).stop
	)

def quicklook(arg):
	pyplot.imshow(arg.state(), vmax=1)
	pyplot.colorbar()
	pyplot.show()

def fill_gaussian(a):
	h, w = a.shape
	y, x = np.ogrid[:h, :w]
	cy, cx = (h - 1) / 2, (w - 1) / 2
	sigma = min(h, w) / 6
	a[:] = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

if __name__ == "__main__":
	solver = solver_donorcell(
		bcx=cyclic,
		bcy=cyclic,
		nx=20,
		ny=30,
	)

	print(solver.i)
	print(solver.j)

	fill_gaussian(solver.state()) 

	print(solver.state())

	solver.courant(0)[:,:] = -.2
	solver.courant(1)[:,:] = .5

	# quicklook(solver)
	# solver.solve(50)
	# quicklook(solver)

	import time

	times = []

	for _ in range(100):
		solver = solver_donorcell(
			bcx=cyclic,
			bcy=cyclic,
			nx=200,
			ny=300,
		)

		fill_gaussian(solver.state()) 

		solver.courant(0)[:,:] = -.2
		solver.courant(1)[:,:] = .5

		start = time.perf_counter()

		solver.solve(500)

		end = time.perf_counter()
		print(f"Time: {end - start:.6f} seconds")

		times.append(end - start)

	print("Average time:")
	print(sum(times)/len(times))