import time

import dolfinx
import pyamg
import pyvista as pv
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.cpp.mesh import to_string
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.io import XDMFFile
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, eigsh, LinearOperator, cg

import gmesh
from gmesh import *
from local_assembler import LocalAssembler
from multiscale import create_kappa, create_point_source, kappa_x, kappa_y, kappa_a2, compute_correction_operator, \
	create_v
from util import read_mesh, interpolation_matrix_non_matching_meshes

proc = MPI.COMM_WORLD.rank


# V
gamma = 1e3
nu = 20.0

num_eigenpairs = 20

def solve_eigenproblem(A, M, num_eigenpairs, without_shift=False):
	if without_shift:
		eigs, vecs = eigsh(A=A, M=M, k=num_eigenpairs, which="SM")
	else:
		ml = pyamg.smoothed_aggregation_solver(A)
		Ainv = LinearOperator(shape=A.shape, matvec=lambda b: cg(A, b, M=ml.aspreconditioner())[0])
		v0 = np.ones(A.shape[0])
		eigs, vecs = eigsh(A,
						   k=num_eigenpairs, M=M,
						   OPinv=Ainv,
						   v0=v0,
						   sigma=0.0, which='LM', return_eigenvectors=True)

	return eigs, vecs


def refine_mesh(msh_c: mesh.Mesh, ct_c: mesh.MeshTags):
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

	return msh_f, ct_f, ft_f, parent_cells


def main(problem: int, num_mesh_refines: int, show_plots: bool):
	t0 = time.time()

	# Create coarse mesh
	#gmesh.create_gmsh(False)
	gmesh.create_simple_gmsh(size=(2, 3))

	# Read coarse mesh
	cell_type = "triangle"  # "quad"
	msh_c, ct_c, ft_c = read_mesh(False, cell_type)

	# Refine coarse mesh to create fine mesh

	msh_f, ct_f, ft_f, parent_cells = refine_mesh(msh_c, ct_c)

	for i in range(num_mesh_refines - 1):
		parent_cells_old = parent_cells.copy()
		msh_f, ct_f, ft_f, parent_cells = refine_mesh(msh_f, ct_f)
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
	v_f = create_v(FS_f, gamma, nu)
	v_c = create_v(FS_c, gamma, nu)

	if problem == 0:
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
		plotter.close()

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
		plotter.close()
	else:
		grid_v_c = pv.UnstructuredGrid(*plot.vtk_mesh(msh_c, msh_c.topology.dim))
		grid_v_c.point_data["v_c"] = v_c.x.array.real
		grid_v_c.set_active_scalars("v_c")
		plotter = pv.Plotter(window_size=[1000, 1000], off_screen=not show_plots)
		plotter.show_axes()
		plotter.show_grid()
		plotter.add_mesh(grid_v_c, show_edges=True, scalars="v_c", cmap="seismic")
		plotter.view_xy()
		if not show_plots:
			plotter.screenshot("data/v_c.png")
		else:
			plotter.show()
		plotter.close()

		grid_v_f = pv.UnstructuredGrid(*plot.vtk_mesh(msh_f, msh_f.topology.dim))
		grid_v_f.point_data["v_f"] = v_f.x.array.real
		grid_v_f.set_active_scalars("v_f")
		plotter = pv.Plotter(window_size=[1000, 1000], off_screen=not show_plots)
		plotter.show_axes()
		plotter.show_grid()
		plotter.add_mesh(grid_v_f, show_edges=True, scalars="v_f", cmap="seismic")
		plotter.view_xy()
		if not show_plots:
			plotter.screenshot("data/v_f.png")
		else:
			plotter.show()
		plotter.close()

	####################################################################################################################

	# Define our problem on fine mesh using Unified Form Language (UFL)
	# and assemble the stiffness and mass matrices A_h and M_h
	# We will be using Backward Euler method for time discretization

	# In essence, only the right-hand side of the equation changes (L),
	# so all other terms we can compute only once and reuse them

	u_f = ufl.TrialFunction(FS_f)
	v_f = ufl.TestFunction(FS_f)
	if problem == 0:
		a = ufl.inner(q * ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx
		m = ufl.inner(u_f, v_f) * ufl.dx

		A_h = fem.assemble_matrix(fem.form(a), bcs_f).to_scipy()
		assert A_h.shape == (num_dofs_f, num_dofs_f)

		fine_stiffness_assembler = LocalAssembler(a)
	else:
		a = ufl.inner(ufl.grad(u_f), ufl.grad(v_f)) * ufl.dx
		m = ufl.inner(u_f, v_f) * ufl.dx

		x = ufl.SpatialCoordinate(msh_f)
		v_expr = gamma * ufl.cos(nu * ufl.pi * (x[0] + 0.1)) * ufl.cos(nu * ufl.pi * x[1])

		vm = ufl.inner(v_expr * u_f, v_f) * ufl.dx
		A_h = fem.assemble_matrix(fem.form(a), bcs_f).to_scipy()
		assert A_h.shape == (num_dofs_f, num_dofs_f)

		VM_h = fem.assemble_matrix(fem.form(vm), bcs_f).to_scipy()
		assert VM_h.shape == (num_dofs_f, num_dofs_f)

		fine_stiffness_assembler = LocalAssembler(a + vm)
		A_h = A_h.copy() + VM_h.copy()

		# Solve the problem in FEniCSx for reference


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

	# We will be interested in certain three modes:
	# Coarse eigenvalues: [3.43, 12.26, 32.37]
	# Fine eigenvalues: [22.36, 13.66, 5.96]
	# LOD eigenvalues: [21.96, 13.14, 6.93]
	coarse_eigenpairs = [["1", 3.43, None], ["3", 12.26, None], ["9", 32.37, None]]
	fine_eigenpairs = [["1", -22.36, None], ["3", -13.66, None], ["9", 5.96, None]]
	lod_eigenpairs = [["1", -21.96, None], ["3", -13.14, None], ["9", 6.93, None]]

	# for i in range(num_steps):
	if problem == 0:
		L = ufl.inner(f, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
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

		# u_n = uhLOD.copy()

		# Save plot
		grid_uhLOD = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
		grid_uhLOD.point_data["uhLOD"] = uhLOD.x.array.real
		grid_uhLOD.set_active_scalars("uhLOD")

		plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
		plotter.show_axes()
		plotter.show_grid()
		plotter.add_mesh(grid_uhLOD, show_edges=True)
		plotter.view_xy()
		plotter.screenshot(f"plot_stationary/uhLOD.png")
		plotter.close()

		with dolfinx.io.XDMFFile(msh_f.comm, "data/uhLOD.xdmf", "w", encoding=XDMFFile.Encoding.ASCII) as xdmf:
			xdmf.write_mesh(msh_f)
			xdmf.write_function(uhLOD)
	else:
		M_H = B_H @ (P_h + Q_h) @ M_h @ (P_h + Q_h).transpose() @ B_H.transpose()
		#VM_H = B_H @ P_h @ VM_h @ P_h.transpose() @ B_H.transpose()

		#is_A_symmetric = np.allclose(A.todense(), A.T.todense(), atol=1e-10)
		#is_M_symmetric = np.allclose(M_H.todense(), M_H.T.todense(), atol=1e-10)
		#assert is_A_symmetric and is_M_symmetric

		eigs, vecs = solve_eigenproblem(A_H_LOD, M_H, num_eigenpairs=100)

		for i in range(len(eigs)):
			r = eigs[i]
			vec = vecs[:, i]
			vec = (P_h + Q_h).transpose() @ B_H.transpose() @ vec

			# Save plot
			grid_uhLOD = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
			grid_uhLOD.point_data["uhLOD"] = vec
			grid_uhLOD.set_active_scalars("uhLOD")

			plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
			plotter.show_axes()
			plotter.show_grid()
			plotter.add_mesh(grid_uhLOD, show_edges=False, scalars="uhLOD", cmap="seismic")
			plotter.view_xy()
			plotter.screenshot(f"plot_stationary/uhLOD_eigen_{r}.png")

			# Save to XDMF
			#u_i_vec = fem.Function(FS_f)
			#u_i_vec.x.array.real = vec
			#with dolfinx.io.XDMFFile(msh_f.comm, f"data/uhLOD_eigen_{r}.xdmf", "w",
			#						 encoding=XDMFFile.Encoding.ASCII) as xdmf:
			#	xdmf.write_mesh(msh_f)
			#	xdmf.write_function(u_i_vec)

			for j in range(len(lod_eigenpairs)):
				if np.isclose(r, lod_eigenpairs[j][1], atol=1e-2):
					lod_eigenpairs[j][2] = vec
					plotter.screenshot(f"plot_to_keep/uhLOD_eigen_mode_{lod_eigenpairs[j][0]}.png")
					break

			plotter.close()

	t02 = time.time()
	print(f"Time to solve LOD system: {t02 - t01}\n")

	####################################################################################################################

	# Solving the system only on coarse mesh for comparison

	t11 = time.time()

	# for i in range(num_steps):

	if problem == 0:
		raise NotImplementedError
	else:
		# Solve the coarse system
		A_H = B_H @ P_h @ A_h @ P_h.transpose() @ B_H.transpose()
		M_H = B_H @ P_h @ M_h @ P_h.transpose() @ B_H.transpose()
		#VM_H = B_H @ P_h @ VM_h @ P_h.transpose() @ B_H.transpose()

		# Check symmetry
		#is_A_H_symmetric = np.allclose(A.todense(), A.T.todense(), atol=1e-10)
		#is_M_H_symmetric = np.allclose(M_H.todense(), M_H.T.todense(), atol=1e-10)
		#assert is_A_H_symmetric and is_M_H_symmetric

		# Solve the eigenvalue problem
		eigs, vecs = solve_eigenproblem(A_H, M_H, num_eigenpairs=100)
		for i in range(len(eigs)):
			r = eigs[i]
			vec = vecs[:, i]
			vec = B_H.transpose() @ vec

			# Save plot
			grid_uh_c = pv.UnstructuredGrid(*plot.vtk_mesh(FS_c))
			grid_uh_c.point_data["uh_c"] = vec
			grid_uh_c.set_active_scalars("uh_c")

			plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
			plotter.show_axes()
			plotter.show_grid()
			plotter.add_mesh(grid_uh_c, show_edges=False, scalars="uh_c", cmap="seismic")
			plotter.view_xy()
			plotter.screenshot(f"plot_stationary/uh_c_eigen_{r}.png")

			# Save to XDMF
			#u_i_vec = fem.Function(FS_c)
			#u_i_vec.x.array.real = vec
			#with dolfinx.io.XDMFFile(msh_c.comm, f"data/uh_c_eigen_{r}.xdmf", "w",
			#						 encoding=XDMFFile.Encoding.ASCII) as xdmf:
			#	xdmf.write_mesh(msh_c)
			#	xdmf.write_function(u_i_vec)

			for j in range(len(coarse_eigenpairs)):
				if np.isclose(r, coarse_eigenpairs[j][1], atol=1e-2):
					coarse_eigenpairs[j][2] = vec
					plotter.screenshot(f"plot_to_keep/uh_c_eigen_mode_{coarse_eigenpairs[j][0]}.png")
					break

			plotter.close()

	t12 = time.time()
	print(f"Time to solve coarse system: {t12 - t11}\n")

	####################################################################################################################
	t20 = time.time()

	# Solving the system only on fine mesh for comparison

	# for i in range(num_steps):
	if problem == 0:
		L = ufl.inner(f, v_f) * ufl.dx  # + dt * ufl.inner(f, v_f) * ufl.dx
		f_h = fem.assemble_vector(fem.form(L)).array
		assert f_h.shape == (num_dofs_f,)

		uh_f = fem.Function(FS_f)
		uh_f.x.array.real = spsolve(A_h, f_h)

		# Save plot
		grid_uh_f = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
		grid_uh_f.point_data["uh_f"] = uh_f.x.array.real
		grid_uh_f.set_active_scalars("uh_f")
		plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
		plotter.show_axes()
		plotter.show_grid()
		plotter.add_mesh(grid_uh_f, show_edges=False, scalars="uh_f")
		plotter.view_xy()
		plotter.screenshot(f"plot_stationary/uh_f.png")
		plotter.close()
	else:
		A = B_h @ A_h @ B_h.transpose() #+ VM_h
		M = B_h @ M_h @ B_h.transpose()
		#is_A_symmetric = np.allclose(A.todense(), A.T.todense(), atol=1e-10)
		#is_M_symmetric = np.allclose(M.todense(), M.T.todense(), atol=1e-10)
		#assert is_A_symmetric and is_M_symmetric

		eigs, vecs = solve_eigenproblem(A, M, num_eigenpairs=100)

		for i in range(len(eigs)):
			r = eigs[i]
			vec = vecs[:, i]
			vec = B_h.transpose() @ vec

			# Save plot
			grid_uh_f = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
			grid_uh_f.point_data["uh_f"] = vec
			grid_uh_f.set_active_scalars("uh_f")

			plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
			plotter.show_axes()
			plotter.show_grid()
			plotter.add_mesh(grid_uh_f, show_edges=False, scalars="uh_f", cmap="seismic")
			plotter.view_xy()
			plotter.screenshot(f"plot_stationary/uh_f_eigen_{r}.png")

			# Save to XDMF
			#u_i_vec = fem.Function(FS_f)
			#u_i_vec.x.array.real = vec
			#with dolfinx.io.XDMFFile(msh_f.comm, f"data/uh_f_eigen_{r}.xdmf", "w",
			#						 encoding=XDMFFile.Encoding.ASCII) as xdmf:
			#	xdmf.write_mesh(msh_f)
			#	xdmf.write_function(u_i_vec)

			for j in range(len(fine_eigenpairs)):
				if np.isclose(r, fine_eigenpairs[j][1], atol=1e-2):
					fine_eigenpairs[j][2] = vec
					plotter.screenshot(f"plot_to_keep/uh_f_eigen_mode_{fine_eigenpairs[j][0]}.png")
					break

			plotter.close()

	t21 = time.time()
	print(f"Time to solve fine system: {t21 - t20}")
	####################################################################################################################

	# Calculate difference between LOD and fine solution and save plots for the three chosen modes
	print("L2 error between LOD and fine solution:")
	for i in range(len(lod_eigenpairs)):
		if lod_eigenpairs[i][2] is not None:
			diff1 = np.abs(lod_eigenpairs[i][2] - fine_eigenpairs[i][2])
			diff2 = np.abs(lod_eigenpairs[i][2] + fine_eigenpairs[i][2])	# it's possible that the modes are opposite
			norm1 = np.linalg.norm(diff1)
			norm2 = np.linalg.norm(diff2)
			if norm1 < norm2:
				diff = diff1
				norm = norm1
			else:
				diff = diff2
				norm = norm2

			grid_diff = pv.UnstructuredGrid(*plot.vtk_mesh(FS_f))
			grid_diff.point_data["diff"] = diff
			grid_diff.set_active_scalars("diff")

			plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
			plotter.show_axes()
			plotter.show_grid()
			plotter.add_mesh(grid_diff, show_edges=False, scalars="diff", cmap="seismic")
			plotter.view_xy()
			plotter.screenshot(f"plot_to_keep/fine-LOD_diff_mode_{lod_eigenpairs[i][0]}.png")
			plotter.close()
			print(f" Mode {lod_eigenpairs[i][0]}: {norm:.4f}")


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
		plotter.close()

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
	plotter.close()
'''

if __name__ == "__main__":
	problems = [
		0,  # elliptic; poisson; heat
		1  # eigenvalue
	]
	main(problem=1, num_mesh_refines=2, show_plots=False)
