import glob
import os
import time

import cv2
import dolfinx
import matplotlib.pyplot as plt
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
from correction_operator import compute_correction_operator
from util_mesh import kappa_x, kappa_a2, kappa_y, create_kappa, create_point_source, read_mesh, \
	interpolation_matrix_non_matching_meshes
from util_pyvista import plot_grid_cells, plot_grid_points, screenshot

proc = MPI.COMM_WORLD.rank

# Time discretization parameters
dt = 0.001
num_steps = 50
T = int(dt * num_steps)
assert num_steps > 0


def main(show_plots: bool):
	t0 = time.time()

	# Create coarse mesh
	create_gmsh((1, 1), False)
	# create_simple_gmsh(cell_marker_1, boundary_marker)

	# Read coarse mesh
	msh_c, ct_c, ft_c = read_mesh(False, "triangle")

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
	plot_grid_cells(msh_f, q.x.array.real, "q", "data/q.png", show_plots, cmap="viridis")
	plot_grid_points(msh_f, f.x.array.real, "f", "data/f.png", show_plots, cmap="viridis")

	# Plot initial state
	if not os.path.exists("plot_solution_LOD"):
		os.makedirs("plot_solution_LOD")
	if not os.path.exists("plot_solution_fine"):
		os.makedirs("plot_solution_fine")
	screenshot(msh_f, f.x.array.real, f"plot_solution_LOD/{'0'.zfill(len(str(num_steps)))}.png", cmap="viridis")
	screenshot(msh_f, f.x.array.real, f"plot_solution_fine/{'0'.zfill(len(str(num_steps)))}.png", cmap="viridis")

	####################################################################################################################
	t0 = time.time()

	# Define our problem on fine mesh using Unified Form Language (UFL)
	# and assemble the stiffness and mass matrices A_h and M_h
	# We will be using Backward Euler method for time discretization

	# In essence, only the right-hand side of the equation changes (L),
	# so all other terms we can compute only once and reuse them

	u_f = ufl.TrialFunction(FS_f)
	v_f = ufl.TestFunction(FS_f)

	a = ufl.inner(u_f, v_f) * ufl.dx + dt * ufl.inner(q * ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx
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

	#####
	uhLODs = []
	uh_fs = []
	#####

	# Time loop
	u_n = f.copy()
	# uhLODs.append(f.copy())

	for i in range(num_steps):
		L = ufl.inner(u_n, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
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
		uhLODs.append(uhLOD.copy())

		# Save plot
		index_str = str(i + 1).zfill(len(str(num_steps)))
		screenshot(msh_f, uhLOD.x.array.real, f"plot_solution_LOD/{index_str}.png")

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
	# uh_fs.append(f.copy())

	u_f = ufl.TrialFunction(FS_f)
	v_f = ufl.TestFunction(FS_f)
	a = ufl.inner(u_f, v_f) * ufl.dx + dt * ufl.inner(q * ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx

	A_h = fem.assemble_matrix(fem.form(a), bcs_f).to_scipy()
	assert A_h.shape == (num_dofs_f, num_dofs_f)
	# M_h = fem.assemble_matrix(fem.form(m), bcs_f).to_scipy()
	# assert M_h.shape == (num_dofs_f, num_dofs_f)

	for i in range(num_steps):
		L = ufl.inner(u_n_f, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
		f_h = fem.assemble_vector(fem.form(L)).array
		assert f_h.shape == (num_dofs_f,)

		uh_f = fem.Function(FS_f)
		uh_f.x.array.real = spsolve(A_h, f_h)

		u_n_f = uh_f.copy()
		uh_fs.append(uh_f.copy())

		# Save plot
		index_str = str(i + 1).zfill(len(str(num_steps)))
		screenshot(msh_f, uh_f.x.array.real, f"plot_solution_fine/{index_str}.png", cmap="viridis")

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

	####################################################################################################################
	# For each time step, calc the diff between LOD and fine solution and plot it and make a video of it too
	# Also, calc the L2-norm in each step and plot it across time

	l2s = []
	for i in range(num_steps):
		uhLOD = uhLODs[i]
		uh_f = uh_fs[i]

		diff = np.abs(uhLOD.x.array.real - uh_f.x.array.real)
		diff_f = fem.Function(FS_f)
		diff_f.x.array.real = diff
		# with dolfinx.io.XDMFFile(msh_f.comm, f"data/solution_diff_{i}.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
		#	xdmf.write_mesh(msh_f)
		#	xdmf.write_function(diff_f)
		index_str = str(i).zfill(len(str(num_steps)))
		screenshot(msh_f, diff_f.x.array.real, f"plot_solution_diff/solution_diff_{index_str}.png", cmap="viridis")

		l2 = np.linalg.norm(diff)
		l2s.append(l2)
	# Save L2-norms across time
	np.savetxt("plot_solution_diff/l2_norms.txt", np.array(l2s))
	# Merge the photos into a nice video
	img_array = []
	for filename in sorted(glob.glob('plot_solution_diff/solution_diff_*.png')):
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width, height)
		img_array.append(img)
	out = cv2.VideoWriter('solution_diff.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()
	# Clean up
	for filename in sorted(glob.glob('plot_solution_diff/solution_diff_*.png')):
		os.remove(filename)

	# Plot L2-norms across time
	plt.figure(figsize=(10, 5))
	plt.plot(np.arange(num_steps), l2s, marker='o')
	plt.title(
		"L2-norma razlike rješenja po vremenskom koraku")  # ("L2-norm of the difference between LOD and fine solution across time")
	plt.xlabel("Vremenski korak")
	plt.ylabel("L2-norma")
	plt.grid()
	plt.savefig("L2.png")
	if show_plots:
		plt.show()
	plt.close()


####################################################################################################################


if __name__ == "__main__":
	main(show_plots=False)
