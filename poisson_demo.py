import time

import dolfinx
import pyvista as pv
import ufl
from dolfinx import fem, plot
from dolfinx.io import XDMFFile
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import gmesh
from gmesh import *
from local_assembler import LocalAssembler
from correction_operator import compute_correction_operator
from util_mesh import create_kappa, create_point_source, kappa_x, kappa_y, kappa_a2, read_mesh, \
	interpolation_matrix_non_matching_meshes, refine_mesh
from util_pyvista import plot_grid_points, plot_grid_cells

proc = MPI.COMM_WORLD.rank


def main(num_mesh_refines: int, show_plots: bool):
	t0 = time.time()

	# Create coarse mesh
	gmesh.create_gmsh((1, 1), False)
	# gmesh.create_simple_gmsh((2, 3))

	# Read coarse mesh
	cell_type = "triangle"  # "quad"
	msh_c, ct_c, ft_c = read_mesh(False, cell_type)

	# Refine coarse mesh to create fine mesh

	msh_f, ct_f, ft_f, parent_cells = refine_mesh(msh_c, ct_c, marker_facet_boundary)

	for i in range(num_mesh_refines - 1):
		parent_cells_old = parent_cells.copy()
		msh_f, ct_f, ft_f, parent_cells = refine_mesh(msh_f, ct_f, marker_facet_boundary)
		for j, c in enumerate(parent_cells):
			parent_cells[j] = parent_cells_old[c]

	# Number of cells on coarse mesh
	num_cells_c = ct_c.indices.shape[0]  # N_T_H
	index_map_c = msh_c.topology.index_map(2)
	assert index_map_c.size_local == index_map_c.size_global
	assert num_cells_c == index_map_c.size_local

	# P1 function space on coarse mesh for solution
	FS_c = fem.functionspace(msh_c, ("CG", 1))
	assert FS_c.dofmap.index_map.size_local == FS_c.dofmap.index_map.size_global
	num_dofs_c = FS_c.dofmap.index_map.size_local * FS_c.dofmap.index_map_bs  # N_H

	# P1 function space on fine mesh for solution
	FS_f = fem.functionspace(msh_f, ("CG", 1))
	assert FS_f.dofmap.index_map.size_local == FS_f.dofmap.index_map.size_global
	num_dofs_f = FS_f.dofmap.index_map.size_local * FS_f.dofmap.index_map_bs  # N_h

	# Define Dirichlet boundary condition on fine mesh
	boundary_facets_f = ft_f.find(marker_facet_boundary)
	boundary_dofs_f = fem.locate_dofs_topological(FS_f, msh_f.topology.dim - 1, boundary_facets_f)
	bcs_f = [fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_f, FS_f)]
	not_boundary_dofs_f = np.setdiff1d(np.arange(num_dofs_f), boundary_dofs_f, assume_unique=True)
	# Create boundary correction (restriction) matrix B_h and boundary condition on fine mesh
	B_h = csr_matrix(
		(
			np.ones_like(not_boundary_dofs_f),
			(
				not_boundary_dofs_f,
				not_boundary_dofs_f
			)
		),
		shape=(num_dofs_f, num_dofs_f)
	)
	assert B_h.shape == (num_dofs_f, num_dofs_f)
	# Remove all-zero rows from B_h
	ind_ptr = np.unique(B_h.indptr)
	B_h = csr_matrix((B_h.data, B_h.indices, ind_ptr), shape=(len(ind_ptr) - 1, B_h.shape[1]))

	# Create boundary correction (restriction) matrix B_H and boundary condition on coarse mesh
	boundary_facets_c = ft_c.find(marker_facet_boundary)
	boundary_dofs_c = fem.locate_dofs_topological(FS_c, 1, boundary_facets_c)
	bcs_c = [fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_c, FS_c)]
	not_boundary_dofs_c = np.setdiff1d(np.arange(num_dofs_c), boundary_dofs_c, assume_unique=True)
	B_H = csr_matrix(
		(
			np.ones_like(not_boundary_dofs_c),
			(
				not_boundary_dofs_c,
				not_boundary_dofs_c
			)
		),
		shape=(num_dofs_c, num_dofs_c)
	)
	assert B_H.shape == (num_dofs_c, num_dofs_c)

	# Remove all-zero rows from B_H
	ind_ptr = np.unique(B_H.indptr)
	B_H = csr_matrix((B_H.data, B_H.indices, ind_ptr), shape=(len(ind_ptr) - 1, B_H.shape[1]))

	# Create kappa and rhs sou_erm f
	q = create_kappa(msh_f, ct_f, marker_cell_outer, marker_cell_inner)[0]
	f = create_point_source(msh_f, kappa_x, kappa_y, kappa_a2)

	# Plot q and point source
	plot_grid_cells(msh_f, q.x.array.real, "q", "data/q.png", show_plots, cmap="viridis")
	plot_grid_points(msh_f, f.x.array.real, "f", "data/f.png", show_plots, cmap="viridis")

	####################################################################################################################

	# Define our problem on fine mesh using Unified Form Language (UFL)
	# and assemble the stiffness and mass matrices A_h and M_h
	# We will be using Backward Euler method for time discretization

	# In essence, only the right-hand side of the equation changes (L),
	# so all other terms we can compute only once and reuse them

	u_f = ufl.TrialFunction(FS_f)
	v_f = ufl.TestFunction(FS_f)

	a = ufl.inner(q * ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx
	m = ufl.inner(u_f, v_f) * ufl.dx

	A_h = fem.assemble_matrix(fem.form(a), bcs_f).to_scipy()
	assert A_h.shape == (num_dofs_f, num_dofs_f)

	fine_stiffness_assembler = LocalAssembler(a)

	M_h = fem.assemble_matrix(fem.form(m), bcs_f).to_scipy()
	assert M_h.shape == (num_dofs_f, num_dofs_f)

	# Create projection matrix P_h from coarse mesh Lagrange space to fine mesh Lagrange space
	P_h = csr_matrix(interpolation_matrix_non_matching_meshes(FS_f, FS_c).transpose())

	# Calculate constraint matrix C_h
	C_h = P_h @ M_h

	t1 = time.time()
	print("Setup time: ", t1 - t0)

	# Create corrector matrix Q_h
	print("Computing correction operator Q...")
	t00 = time.time()
	Q_h = compute_correction_operator(msh_c, ct_c, parent_cells,
									  FS_c, FS_f,
									  boundary_dofs_c, boundary_dofs_f,
									  A_h, P_h, C_h,
									  fine_stiffness_assembler)
	t01 = time.time()
	print(f"Time to compute correction operator Q: {t01 - t00}")
	#        t0 = time.time()

	A_H_LOD = B_H @ (P_h + Q_h) @ A_h @ (P_h + Q_h).transpose() @ B_H.transpose()

	L = ufl.inner(f, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
	f_h = fem.assemble_vector(fem.form(L)).array
	assert f_h.shape == (num_dofs_f,)

	# Add the corrector matrix to the solution and solve the system
	f_H_LOD = B_H @ (P_h + Q_h) @ f_h
	u_H_LOD = B_H.transpose() @ spsolve(A_H_LOD, f_H_LOD)
	u_h_LOD = (P_h + Q_h).transpose() @ u_H_LOD

	t02 = time.time()
	print(f"Time to solve LOD system: {t02 - t01}\n")

	# u_h_LOD is now a vector of size num_dofs_f, which we can wrap in a Function object using dolfinx
	# and display using pyvista or ParaView
	uhLOD = fem.Function(FS_f)
	uhLOD.x.array.real = u_h_LOD  # spsolve(A_h, f_h)

	# u_n = uhLOD.copy()

	# Save plot
	plot_grid_points(msh_f, uhLOD.x.array.real, "u_LOD", "plot_poisson/u_LOD.png", show_plots, cmap="viridis", show_edges=False)

	with dolfinx.io.XDMFFile(msh_f.comm, "data/u_LOD.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
		xdmf.write_mesh(msh_f)
		xdmf.write_function(uhLOD)

	####################################################################################################################

	# Solving the system only on coarse mesh for comparison

	t11 = time.time()
	# Coarse solution for comparison
	A_H = B_H @ P_h @ A_h @ P_h.transpose() @ B_H.transpose()
	f_H = B_H @ P_h @ f_h
	uh_c = fem.Function(FS_c)
	uh_c.x.array.real = B_H.transpose() @ spsolve(A_H, f_H)

	t12 = time.time()
	print(f"Time to solve coarse system: {t12 - t11}\n")

	# Save plot
	plot_grid_points(msh_c, uh_c.x.array.real, "u_c", "plot_poisson/u_c.png", show_plots, cmap="viridis", show_edges=False)

	grid_uh_c = pv.UnstructuredGrid(*plot.vtk_mesh(FS_c))
	grid_uh_c.point_data["u_c"] = uh_c.x.array.real
	grid_uh_c.set_active_scalars("u_c")

	# Interpolate coarse solution to fine mesh
	# https://github.com/FEniCS/dolfinx/blob/v0.8.0/python/test/unit/fem/test_interpolation.py
	uh_c_to_f = fem.Function(FS_f)
	uh_c_to_f.interpolate(uh_c, nmm_interpolation_data=dolfinx.fem.create_nonmatching_meshes_interpolation_data(
		uh_c_to_f.function_space.mesh._cpp_object,
		uh_c_to_f.function_space.element,
		uh_c.function_space.mesh._cpp_object,
		padding=1e-14
	))

	####################################################################################################################
	t20 = time.time()

	# Solving the system only on fine mesh for comparison
	L = ufl.inner(f, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
	f_h = fem.assemble_vector(fem.form(L)).array
	assert f_h.shape == (num_dofs_f,)

	uh_f = fem.Function(FS_f)
	uh_f.x.array.real = spsolve(A_h, f_h)

	t21 = time.time()
	print(f"Time to solve fine system: {t21 - t20}")

	# Save plot
	plot_grid_points(msh_f, uh_f.x.array.real, "u_f", "plot_poisson/u_f.png", show_plots, cmap="viridis", show_edges=False)
	####################################################################################################################

	# Calculate L2 errors
	L2_c_f = fem.assemble_scalar(fem.form(
		ufl.inner(uh_c_to_f - uh_f, uh_c_to_f - uh_f) * ufl.dx
	))
	L2_c_f = np.sqrt(msh_f.comm.allreduce(L2_c_f, op=MPI.SUM))

	L2_LOD_c = fem.assemble_scalar(fem.form(
		ufl.inner(uhLOD - uh_c_to_f, uhLOD - uh_c_to_f) * ufl.dx
	))
	L2_LOD_c = np.sqrt(msh_f.comm.allreduce(L2_LOD_c, op=MPI.SUM))

	L2_LOD_f = fem.assemble_scalar(fem.form(
		ufl.inner(uhLOD - uh_f, uhLOD - uh_f) * ufl.dx
	))
	L2_LOD_f = np.sqrt(msh_f.comm.allreduce(L2_LOD_f, op=MPI.SUM))

	print(f"L2 error between coarse and fine solution: {L2_c_f}")
	print(f"L2 error between LOD and coarse solution: {L2_LOD_c}")
	print(f"L2 error between LOD and fine solution: {L2_LOD_f}")

	# Plot absolute difference between LOD and fine solution
	diff = fem.Function(FS_f)
	diff.x.array.real = uhLOD.x.array.real - uh_f.x.array.real
	for i in range(diff.x.array.real.size):
		if diff.x.array.real[i] < 0.0:
			diff.x.array.real[i] = -diff.x.array.real[i]
	grid_diff = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
	grid_diff.point_data["diff"] = diff.x.array.real
	grid_diff.set_active_scalars("diff")

	with dolfinx.io.XDMFFile(msh_f.comm, "data/solution_diff.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
		xdmf.write_mesh(msh_f)
		xdmf.write_function(diff)

	plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
	plotter.show_axes()
	plotter.show_grid()
	plotter.add_mesh(grid_diff, show_edges=True)
	plotter.view_xy()
	plotter.screenshot("plot_poisson/solution_diff.png")
	plotter.close()


if __name__ == "__main__":
	main(num_mesh_refines=2, show_plots=False)
