# implementation based on Arabas et al. 2014 https://doi.org/10.3233/SPR-140379

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

real_t = "float32"

class Shift():
	def __init__(self, plus, minus):
		self.plus = plus
		self.minus = minus
	def __radd__(self, arg): 
		return type(arg)(
			arg.start + self.plus, 
			arg.stop + self.plus
		)
	def __rsub__(self, arg): 
		return type(arg)(
			arg.start - self.minus, 
			arg.stop - self.minus
		)

one = Shift(1, 1)
half = Shift(0, 1)

def pi(d, *idx): 
	return (idx[d], idx[d-1])

@jax.jit
def f(psi_l: jax.Array, psi_r: jax.Array, C: jax.Array):
	return (
		(C + jnp.abs(C)) * psi_l + 
		(C - jnp.abs(C)) * psi_r
	) / 2
	
def donorcell(d, psi, C, i, j):
	return (
		f(
		    psi[pi(d, i, j)], 
		    psi[pi(d, i+one, j)], 
			C[pi(d, i+half, j)]
		) - 
		f(
		    psi[pi(d, i-one, j)], 
		    psi[pi(d, i, j)], 
			C[pi(d, i-half, j)]
		) 
	)

class BoundaryCondition(ABC):
	@abstractmethod
	def __init__(self, d, i, hlo):
		pass

	@abstractmethod
	def fill_halos(self, psi, j) -> jax.Array:
		pass

class Cyclic(BoundaryCondition):
	def __init__(self, d, i, halo): 
		self.d = d
		self.left_halo = slice(i.start - halo, i.start)
		self.right_edge = slice(i.stop - halo, i.stop)
		self.right_halo = slice(i.stop, i.stop + halo)
		self.left_edge = slice(i.start, i.start + halo)

	def fill_halos(self, psi: jax.Array, j: int) -> jax.Array:
		psi = psi.at[pi(self.d, self.left_halo, j)].set(
				psi[pi(self.d, self.right_edge, j)]
			)
		psi = psi.at[pi(self.d, self.right_halo, j)].set(
			psi[pi(self.d, self.left_edge, j)]
		)

		return psi

class Solver(ABC):
	def __init__(self, bcx: type[BoundaryCondition], bcy: type[BoundaryCondition], nx: int, ny: int, halo: int):
		self.n = 0
		self.halo = halo
		self.i = slice(halo, nx + halo)
		self.j = slice(halo, ny + halo)

		self.bcx = bcx(0, self.i, halo)
		self.bcy = bcy(1, self.j, halo)

		self.psi = [
			jnp.empty((
				ext(self.i, self.halo).stop, 
				ext(self.j, self.halo).stop
			), real_t),
			jnp.empty((
				ext(self.i, self.halo).stop, 
				ext(self.j, self.halo).stop
			), real_t)
		]

		self.C = [
			jnp.empty((
				ext(self.i, half).stop, 
				ext(self.j, self.halo).stop
			), real_t),
			jnp.empty((
				ext(self.i, self.halo).stop, 
				ext(self.j, half).stop
			), real_t)
		]

	def state(self):
		return self.psi[self.n][self.i, self.j]
	
	def set_state(self, a: jax.Array):
		self.psi[self.n] = self.psi[self.n].at[self.i, self.j].set(a)

	def courant(self,d):
		return self.C[d][:]
	
	def set_courant(self, d, a):
		self.C[d] = self.C[d].at[:, :].set(a)

	def cycle(self):
		self.n  = (self.n + 1) % 2 - 2

	@abstractmethod
	def advop(self):
		pass

	def solve(self, nt):
		for t in range(nt):
			self.psi[self.n] = self.bcx.fill_halos(
				self.psi[self.n], ext(self.j, self.halo)
			)
			self.psi[self.n] = self.bcy.fill_halos(
				self.psi[self.n], ext(self.i, self.halo)
			)
			self.advop() 
			self.cycle()

def donorcell_op(psi, n, C, i, j):
	return psi[n].at[i,j].subtract(
		donorcell(0, psi[n], C[0], i, j) +
		donorcell(1, psi[n], C[1], j, i)
	)

class SolverDonorcell(Solver):
	def __init__(self, bcx: type[BoundaryCondition], bcy: type[BoundaryCondition], nx: int, ny: int):
		super().__init__(bcx, bcy, nx, ny, 1)

	def advop(self):
		self.psi[self.n + 1] = self.psi[self.n].at[self.i, self.j].subtract(
			donorcell(0, self.psi[self.n], self.C[0], self.i, self.j) +
			donorcell(1, self.psi[self.n], self.C[1], self.j, self.i)
		)

def quicklook(arg):
	fig, ax = plt.subplots()
	
	im = ax.imshow(arg, vmax=1)
	fig.colorbar(im, ax=ax)

@jax.jit
def fill_gaussian(a: jax.Array) -> jax.Array:
	h, w = a.shape
	y, x = jnp.ogrid[:h, :w]
	cy, cx = (h - 1) / 2, (w - 1) / 2
	sigma = min(h, w) / 6
	return jnp.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))

def ext(r, n):
	if (type(n) == int) & (n == 1): 
		n = one
	return slice(
		(r - n).start, 
		(r + n).stop
	)

i_idxs =  slice(1, 21, None)
j_idxs = slice(1, 31, None)

solver = SolverDonorcell(
	bcx=Cyclic,
	bcy=Cyclic,
	nx=20,
	ny=30,
)

solver.set_courant(0, -0.2)
solver.set_courant(1, 0.5)

solver.set_state(fill_gaussian(solver.state()))

quicklook(solver.state())
solver.solve(50)
quicklook(solver.state())

plt.show()