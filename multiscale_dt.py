import glob
import os
import time

import cv2
import dolfinx
import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.cpp.mesh import to_string
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.io import XDMFFile
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from gmesh import *
from local_assembler import LocalAssembler
from multiscale import kappa_x, kappa_a2, kappa_y, create_kappa, create_point_source, compute_correction_operator
from util import read_mesh, interpolation_matrix_non_matching_meshes

proc = MPI.COMM_WORLD.rank

# Time discretization parameters
dt = 0.001
num_steps = 50
T = int(dt * num_steps)
assert num_steps > 0


def main(problem: int, show_plots: bool):
	t0 = time.time()

	# Create coarse mesh
	create_gmsh(False)
	# create_simple_gmsh(cell_marker_1, boundary_marker)

	# Read coarse mesh
	msh_c, ct_c, ft_c = read_mesh(False)

	# Create fine mesh
	# https://fenicsproject.discourse.group/t/input-for-mesh-refinement-with-refine-plaza/13426/4
	msh_f, parent_cells, _ = mesh.refine_plaza(
		msh_c,
		edges=None,  # Mesh refinement is uniform
		redistribute=False,  # Do not redistribute across processes
		option=RefinementOption.parent_cell  # We are interested in parent cells and parent facets
	)

	# Fix parent cells
	marker = np.arange(msh_c.topology.index_map(msh_c.topology.dim).size_local, dtype=np.int32)
	ct_parent_c = dolfinx.mesh.meshtags(msh_c, msh_c.topology.dim, marker, marker)
	ct_parent_f = dolfinx.mesh.MeshTags(
		dolfinx.cpp.refinement.transfer_cell_meshtag(ct_parent_c._cpp_object, msh_f.topology, parent_cells)
	)

	# Transfer cell mesh tags from coarse to fine mesh
	msh_f.topology.create_connectivity(msh_f.topology.dim, msh_f.topology.dim - 1)
	ct_f = dolfinx.mesh.MeshTags(
		dolfinx.cpp.refinement.transfer_cell_meshtag(ct_c._cpp_object, msh_f.topology, parent_cells)
	)

	# Setup parent cells to reuse later
	parent_cells = []
	for i in range(ct_parent_f.indices.shape[0]):
		cell = ct_parent_f.indices[i]
		value = ct_parent_f.values[i]
		parent_cells.insert(cell, value)
	parent_cells = np.array(parent_cells)

	# Make new facet tags on fine mesh (until we figure out why transferring doesn't work)
	facets = mesh.locate_entities_boundary(msh_f, msh_f.topology.dim - 1, lambda x: np.full(x.shape[1], True))
	ft_f = mesh.meshtags(msh_f, 1, facets, marker_facet_boundary)

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

	# Create boundary correction (restriction) matrix B_H
	boundary_facets_c = ft_c.find(marker_facet_boundary)
	boundary_dofs_c = fem.locate_dofs_topological(FS_c, 1, boundary_facets_c)
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

	# Create kappa and rhs source term f
	q = create_kappa(msh_f, ct_f, marker_cell_outer, marker_cell_inner)[0]
	f = create_point_source(msh_f, kappa_x, kappa_y, kappa_a2)

	# Plot q and point source
	grid_q = pv.UnstructuredGrid(*plot.vtk_mesh(msh_f, msh_f.topology.dim))
	grid_q.cell_data["q"] = q.x.array.real
	grid_q.set_active_scalars("q")
	plotter = pv.Plotter(window_size=[1000, 1000], off_screen=not show_plots)
	plotter.show_axes()
	plotter.show_grid()
	plotter.add_mesh(grid_q, show_edges=True)
	plotter.view_xy()
	if not show_plots:
		plotter.screenshot("data/q.png")
	else:
		plotter.show()

	grid_ps = pv.UnstructuredGrid(*plot.vtk_mesh(msh_f, msh_f.topology.dim))
	grid_ps.point_data["f"] = f.x.array.real
	grid_ps.set_active_scalars("f")
	plotter = pv.Plotter(window_size=[1000, 1000], off_screen=not show_plots)
	plotter.show_axes()
	plotter.show_grid()
	plotter.add_mesh(grid_ps, show_edges=True)
	plotter.view_xy()
	if not show_plots:
		plotter.screenshot("data/f.png")
	else:
		plotter.show()

	# Plot initial state
	grid_f = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
	grid_f.point_data["uhLOD"] = f.x.array.real
	grid_f.set_active_scalars("uhLOD")
	plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
	plotter.show_axes()
	plotter.show_grid()
	plotter.add_mesh(grid_f, show_edges=True)
	plotter.view_xy()
	plotter.screenshot(f"plot_solution_LOD/{"0".zfill(len(str(num_steps)))}.png")
	plotter.screenshot(f"plot_solution_fine/{"0".zfill(len(str(num_steps)))}.png")

	####################################################################################################################
	t0 = time.time()

	# Define our problem on fine mesh using Unified Form Language (UFL)
	# and assemble the stiffness and mass matrices A_h and M_h
	# We will be using Backward Euler method for time discretization

	# In essence, only the right-hand side of the equation changes (L),
	# so all other terms we can compute only once and reuse them

	u_f = ufl.TrialFunction(FS_f)
	v_f = ufl.TestFunction(FS_f)
	if problem == 0:
		a = ufl.inner(u_f, v_f) * ufl.dx + dt * ufl.inner(q * ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx
		m = ufl.inner(u_f, v_f) * ufl.dx
	else:
		v_expr = q.copy()
		a = ufl.inner(u_f, v_f) * ufl.dx + dt * ufl.inner(ufl.grad(u_f),
														  ufl.grad(v_f)) * ufl.dx + dt * v_expr * u_f * v_f * ufl.dx
		m = ufl.inner(u_f, v_f) * ufl.dx
	fine_stiffness_assembler = LocalAssembler(a)

	A_h = fem.assemble_matrix(fem.form(a), bcs_f).to_scipy()
	assert A_h.shape == (num_dofs_f, num_dofs_f)
	M_h = fem.assemble_matrix(fem.form(m), bcs_f).to_scipy()
	assert M_h.shape == (num_dofs_f, num_dofs_f)

	# Create projection matrix P_h from coarse mesh Lagrange space to fine mesh Lagrange space
	P_h = csr_matrix(interpolation_matrix_non_matching_meshes(FS_f, FS_c).transpose())

	# Calculate constraint matrix C_h
	C_h = P_h @ M_h

	#        print(f"Setup time: {time.time() - t0}")
	#        t0 = time.time()

	# Create corrector matrix Q_h
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

	# Time loop
	u_n = f.copy()

	for i in range(num_steps):
		if problem == 0:
			L = ufl.inner(u_n, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
		else:
			L = ufl.inner(u_n, v_f) * ufl.dx  # + dt * ufl.inner(l * u_n, v_f) * ufl.dx
		f_h = fem.assemble_vector(fem.form(L)).array
		assert f_h.shape == (num_dofs_f,)

		# Add the corrector matrix to the solution and solve the system
		f_H_LOD = B_H @ (P_h + Q_h) @ f_h
		u_H_LOD = B_H.transpose() @ spsolve(A_H_LOD, f_H_LOD)
		u_h_LOD = (P_h + Q_h).transpose() @ u_H_LOD

		#        print(f"Time to solve LOD system: {time.time() - t0}")
		#        t0 = time.time()

		# u_h_LOD is now a vector of size num_dofs_f, which we can wrap in a Function object using dolfinx
		# and display using pyvista or ParaView
		uhLOD = fem.Function(FS_f)
		uhLOD.x.array.real = u_h_LOD  # spsolve(A_h, f_h)

		u_n = uhLOD.copy()

		# Save plot
		grid_uhLOD = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
		grid_uhLOD.point_data["uhLOD"] = uhLOD.x.array.real
		grid_uhLOD.set_active_scalars("uhLOD")

		plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
		plotter.show_axes()
		plotter.show_grid()
		plotter.add_mesh(grid_uhLOD, show_edges=True)
		plotter.view_xy()
		index_str = str(i + 1).zfill(len(str(num_steps)))
		plotter.screenshot(f"plot_solution_LOD/{index_str}.png")

	t1 = time.time()
	####################################################################################################################
	print(f"Time to solve the whole LOD system: {t1 - t0}")

	grid_uhLOD = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
	grid_uhLOD.point_data["uhLOD"] = uhLOD.x.array.real
	grid_uhLOD.set_active_scalars("uhLOD")

	with dolfinx.io.XDMFFile(msh_f.comm, "data/solution_LOD.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
		xdmf.write_mesh(msh_f)
		xdmf.write_function(uhLOD)

	# Merge the photos into a nice video
	img_array = []
	for filename in sorted(glob.glob('plot_solution_LOD/*.png')):
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width, height)
		img_array.append(img)

	out = cv2.VideoWriter('solution_LOD.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

	# Clean up
	for filename in sorted(glob.glob('plot_solution_LOD/*.png')):
		os.remove(filename)

	####################################################################################################################
	t0 = time.time()

	# Solving the system only on fine mesh for comparison

	u_n_f = f.copy()

	u_f = ufl.TrialFunction(FS_f)
	v_f = ufl.TestFunction(FS_f)
	if problem == 0:
		a = ufl.inner(u_f, v_f) * ufl.dx + dt * ufl.inner(q * ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx
	# m = ufl.inner(u_f, v_f) * ufl.dx
	else:
		v_expr = q.copy()
		a = ufl.inner(u_f, v_f) * ufl.dx + dt * ufl.inner(ufl.grad(u_f),
														  ufl.grad(v_f)) * ufl.dx + dt * v_expr * u_f * v_f * ufl.dx
	# m = ufl.inner(u_f, v_f) * ufl.dx

	A_h = fem.assemble_matrix(fem.form(a), bcs_f).to_scipy()
	assert A_h.shape == (num_dofs_f, num_dofs_f)
	# M_h = fem.assemble_matrix(fem.form(m), bcs_f).to_scipy()
	# assert M_h.shape == (num_dofs_f, num_dofs_f)

	for i in range(num_steps):
		if problem == 0:
			L = ufl.inner(u_n_f, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
		else:
			L = ufl.inner(u_n_f, v_f) * ufl.dx  # + dt * ufl.inner(l * u_n, v_f) * ufl.dx
		f_h = fem.assemble_vector(fem.form(L)).array
		assert f_h.shape == (num_dofs_f,)

		uh_f = fem.Function(FS_f)
		uh_f.x.array.real = spsolve(A_h, f_h)

		u_n_f = uh_f.copy()

		# Save plot
		grid_uh_f = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
		grid_uh_f.point_data["uh_f"] = uh_f.x.array.real
		grid_uh_f.set_active_scalars("uh_f")
		plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
		plotter.show_axes()
		plotter.show_grid()
		plotter.add_mesh(grid_uh_f, show_edges=True)
		plotter.view_xy()
		index_str = str(i + 1).zfill(len(str(num_steps)))
		plotter.screenshot(f"plot_solution_fine/{index_str}.png")

	t1 = time.time()
	####################################################################################################################
	print(f"Time to solve fine system: {t1 - t0}")

	# Merge the photos into a nice video
	img_array = []
	for filename in sorted(glob.glob('plot_solution_fine/*.png')):
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width, height)
		img_array.append(img)
	out = cv2.VideoWriter('solution_fine.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

	# Clean up
	for filename in sorted(glob.glob('plot_solution_fine/*.png')):
		os.remove(filename)


'''        
    # Coarse solution for comparison
    A_H = B_H @ P_h @ A_h @ P_h.transpose() @ B_H.transpose()
    f_H = B_H @ P_h @ f_h
    uh_c = fem.Function(FS_c)
    uh_c.x.array.real = B_H.transpose() @ spsolve(A_H, f_H)
    print(f"Time to solve just the coarse system: {time.time() - t0}")
    t0 = time.time()
    grid_uh_c = pv.UnstructuredGrid(*plot.vtk_mesh(FS_c))
    grid_uh_c.point_data["uh_c"] = uh_c.x.array.real
    grid_uh_c.set_active_scalars("uh_c")

    with dolfinx.io.XDMFFile(msh_c.comm, "data/solution_coarse.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh_c)
        xdmf.write_function(uh_c)

    # Fine solution for comparison
    uh_f = fem.Function(FS_f)
    uh_f.x.array.real = spsolve(A_h, f_h)
    print(f"Time to solve whole fine system: {time.time() - t0}")
    grid_uh_f = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
    grid_uh_f.point_data["uh_f"] = uh_f.x.array.real
    grid_uh_f.set_active_scalars("uh_f")

    with dolfinx.io.XDMFFile(msh_f.comm, "data/solution_fine.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
        xdmf.write_mesh(msh_f)
        xdmf.write_function(uh_f)

    # pv.start_xvfb()
    for grid, suffix in [(grid_uh_c, "coarse"), (grid_uh_f, "fine"), (grid_uhLOD, "LOD")]:
        plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
        plotter.show_axes()
        plotter.show_grid()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.screenshot(f"data/solution_{suffix}.png")

    # Interpolate coarse solution to fine mesh
    # https://github.com/FEniCS/dolfinx/blob/v0.8.0/python/test/unit/fem/test_interpolation.py
    uh_c_to_f = fem.Function(FS_f)
    uh_c_to_f.interpolate(uh_c, nmm_interpolation_data=dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        uh_c_to_f.function_space.mesh._cpp_object,
        uh_c_to_f.function_space.element,
        uh_c.function_space.mesh._cpp_object,
        padding=1e-14
    ))

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
    plotter.screenshot("data/solution_diff.png")
'''

if __name__ == "__main__":
	problems = [
		0,  # elliptic; poisson; heat
		1  # eigenvalue
	]
	main(problem=0, show_plots=True)
