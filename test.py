import numpy as np
import pyvista as pv
import ufl
from dolfinx import mesh, fem, plot
from mpi4py import MPI
from scipy.sparse import csr_matrix

from multiscale_stationary import solve_eigenproblem


def main():

	# Create mesh on domain (0,2) x (0,3)
	domain = mesh.create_rectangle(
		MPI.COMM_WORLD,
		points=[[0.0, 0.0], [2.0, 3.0]],
		n=[40, 60],  # mesh resolution
		cell_type=mesh.CellType.triangle
	)

	# Function space (P1)
	V = fem.FunctionSpace(domain, ("CG", 1))

	# Define Dirichlet BC: u = 0 on ∂Ω
	u_bc = fem.Function(V)
	u_bc.x.array[:] = 0.0
	facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1,
										   lambda x: np.full(x.shape[1], True))
	dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
	bcs = [fem.dirichletbc(u_bc, dofs)]

	# Create boundary correction (restriction) matrix B
	num_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
	not_boundary_dofs_f = np.setdiff1d(np.arange(num_dofs), dofs, assume_unique=True)
	B = csr_matrix(
		(
			np.ones_like(not_boundary_dofs_f),
			(
				not_boundary_dofs_f,
				not_boundary_dofs_f
			)
		),
		shape=(num_dofs, num_dofs)
	)
	assert B.shape == (num_dofs, num_dofs)
	# Remove all-zero rows from B_h
	ind_ptr = np.unique(B.indptr)
	B = csr_matrix((B.data, B.indices, ind_ptr), shape=(len(ind_ptr) - 1, B.shape[1]))

	# Define trial and test functions
	u = ufl.TrialFunction(V)
	v = ufl.TestFunction(V)

	# Define potential V(x)
	x = ufl.SpatialCoordinate(domain)
	nu = 10
	gamma = 1e3
	V_expr = gamma * ufl.cos(nu * ufl.pi * (x[0] + 0.1)) * ufl.cos(nu * ufl.pi * x[1])

	# Bilinear forms
	a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx #+ ufl.inner(V_expr * u, v) * ufl.dx  # stiffness
	m = ufl.inner(u, v) * ufl.dx  # mass

	# Assemble matrices
	A = fem.assemble_matrix(fem.form(a), bcs=bcs).to_scipy()
	M = fem.assemble_matrix(fem.form(m), bcs=bcs).to_scipy()

	# Solve
	eigs, vecs = solve_eigenproblem(B @ A @ B.transpose(), B @ M @ B.transpose(), 1)

	for i in range(len(eigs)):
		eigval = eigs[i]
		eigvec = vecs[:, i]
		eigvec = B.transpose() @ eigvec
		print(f"Eigenvalue {i}: {eigval:.6f}")

		# Plot
		grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
		grid.point_data["u"] = eigvec
		grid.set_active_scalars("u")

		plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
		plotter.show_axes()
		plotter.show_grid()
		plotter.add_mesh(grid, scalars="u", show_edges=True, cmap="seismic")
		plotter.view_xy()
		plotter.screenshot(f"plot_test/g={gamma}_nu={nu}_eigen_{eigval}.png")


if __name__ == "__main__":
	main()
