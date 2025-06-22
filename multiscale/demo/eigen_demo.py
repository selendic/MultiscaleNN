import os
import time

import dolfinx
import pyamg
import ufl
from dolfinx import fem
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, cg

import gmesh
from correction_operator import compute_correction_operator
from gmesh import *
from local_assembler import LocalAssembler
from multiscale.util.util_mesh import create_v, read_mesh, refine_mesh, interpolation_matrix_non_matching_meshes
from multiscale.util.util_pyvista import plot_grid_points, screenshot

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


def main(num_mesh_refines: int, show_plots: bool):
	t0 = time.time()

	# Create coarse mesh
	# gmesh.create_gmsh(False)
	gmesh.create_simple_gmsh((2, 3))

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
	v_f = create_v(FS_f, gamma, nu)
	v_c = create_v(FS_c, gamma, nu)

	# Plot the coefficient
	if not os.path.exists("data"):
		os.makedirs("data")
	if not os.path.exists("plot_eigen"):
		os.makedirs("plot_eigen")
	if not os.path.exists("plot_to_keep"):
		os.makedirs("plot_to_keep")
	plot_grid_points(msh_c, v_c.x.array.real, "v_c", "data/v_c.png", show_plots, cmap="seismic")
	plot_grid_points(msh_f, v_f.x.array.real, "v_f", "data/v_f.png", show_plots, cmap="seismic")


	####################################################################################################################

	# Define our problem on fine mesh using Unified Form Language (UFL)
	# and assemble the stiffness and mass matrices A_h and M_h
	# We will be using Backward Euler method for time discretization

	# In essence, only the right-hand side of the equation changes (L),
	# so all other terms we can compute only once and reuse them

	u_f = ufl.TrialFunction(FS_f)
	v_f = ufl.TestFunction(FS_f)

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

	M_H = B_H @ (P_h + Q_h) @ M_h @ (P_h + Q_h).transpose() @ B_H.transpose()
	# VM_H = B_H @ P_h @ VM_h @ P_h.transpose() @ B_H.transpose()

	# is_A_symmetric = np.allclose(A.todense(), A.T.todense(), atol=1e-10)
	# is_M_symmetric = np.allclose(M_H.todense(), M_H.T.todense(), atol=1e-10)
	# assert is_A_symmetric and is_M_symmetric

	eigs, vecs = solve_eigenproblem(A_H_LOD, M_H, num_eigenpairs=100)

	t02 = time.time()
	print(f"Time to solve LOD system: {t02 - t01}\n")

	for i in range(len(eigs)):
		r = eigs[i]
		vec = vecs[:, i]
		vec = (P_h + Q_h).transpose() @ B_H.transpose() @ vec

		# Save plot
		plot_grid_points(msh_f, vec, "u_LOD", f"plot_eigen/u_LOD_eigen_{r}.png", False, cmap="seismic",
						 show_edges=False)

		# Save to XDMF
		# u_i_vec = fem.Function(FS_f)
		# u_i_vec.x.array.real = vec
		# with dolfinx.io.XDMFFile(msh_f.comm, f"data/uhLOD_eigen_{r}.xdmf", "w",
		#						 encoding=XDMFFile.Encoding.ASCII) as xdmf:
		#	xdmf.write_mesh(msh_f)
		#	xdmf.write_function(u_i_vec)

		for j in range(len(lod_eigenpairs)):
			if np.isclose(r, lod_eigenpairs[j][1], atol=1e-2):
				lod_eigenpairs[j][2] = vec
				screenshot(msh_f, vec, f"plot_to_keep/u_LOD_eigen_mode_{lod_eigenpairs[j][0]}.png")
				break

	####################################################################################################################

	# Solving the system only on coarse mesh for comparison

	t11 = time.time()

	# Solve the coarse system
	A_H = B_H @ P_h @ A_h @ P_h.transpose() @ B_H.transpose()
	M_H = B_H @ P_h @ M_h @ P_h.transpose() @ B_H.transpose()
	# VM_H = B_H @ P_h @ VM_h @ P_h.transpose() @ B_H.transpose()

	# Check symmetry
	# is_A_H_symmetric = np.allclose(A.todense(), A.T.todense(), atol=1e-10)
	# is_M_H_symmetric = np.allclose(M_H.todense(), M_H.T.todense(), atol=1e-10)
	# assert is_A_H_symmetric and is_M_H_symmetric

	# Solve the eigenvalue problem
	eigs, vecs = solve_eigenproblem(A_H, M_H, num_eigenpairs=100)

	t12 = time.time()
	print(f"Time to solve coarse system: {t12 - t11}\n")

	for i in range(len(eigs)):
		r = eigs[i]
		vec = vecs[:, i]
		vec = B_H.transpose() @ vec

		# Save plot
		plot_grid_points(msh_c, vec, "u_c", f"plot_eigen/u_c_eigen_{r}.png", False, cmap="seismic", show_edges=False)

		# Save to XDMF
		# u_i_vec = fem.Function(FS_c)
		# u_i_vec.x.array.real = vec
		# with dolfinx.io.XDMFFile(msh_c.comm, f"data/uh_c_eigen_{r}.xdmf", "w",
		#						 encoding=XDMFFile.Encoding.ASCII) as xdmf:
		#	xdmf.write_mesh(msh_c)
		#	xdmf.write_function(u_i_vec)

		for j in range(len(coarse_eigenpairs)):
			if np.isclose(r, coarse_eigenpairs[j][1], atol=1e-2):
				coarse_eigenpairs[j][2] = vec
				screenshot(msh_c, vec, f"plot_to_keep/u_c_eigen_mode_{coarse_eigenpairs[j][0]}.png")
				break

	####################################################################################################################
	t20 = time.time()

	# Solving the system only on fine mesh for comparison

	A = B_h @ A_h @ B_h.transpose()  # + VM_h
	M = B_h @ M_h @ B_h.transpose()
	# is_A_symmetric = np.allclose(A.todense(), A.T.todense(), atol=1e-10)
	# is_M_symmetric = np.allclose(M.todense(), M.T.todense(), atol=1e-10)
	# assert is_A_symmetric and is_M_symmetric

	eigs, vecs = solve_eigenproblem(A, M, num_eigenpairs=100)

	t21 = time.time()
	print(f"Time to solve fine system: {t21 - t20}")

	for i in range(len(eigs)):
		r = eigs[i]
		vec = vecs[:, i]
		vec = B_h.transpose() @ vec

		# Save plot
		plot_grid_points(msh_f, vec, "u_f", f"plot_eigen/u_f_eigen_{r}.png", False, cmap="seismic", show_edges=False)

		# Save to XDMF
		# u_i_vec = fem.Function(FS_f)
		# u_i_vec.x.array.real = vec
		# with dolfinx.io.XDMFFile(msh_f.comm, f"data/uh_f_eigen_{r}.xdmf", "w",
		#						 encoding=XDMFFile.Encoding.ASCII) as xdmf:
		#	xdmf.write_mesh(msh_f)
		#	xdmf.write_function(u_i_vec)

		for j in range(len(fine_eigenpairs)):
			if np.isclose(r, fine_eigenpairs[j][1], atol=1e-2):
				fine_eigenpairs[j][2] = vec
				screenshot(msh_f, vec, f"plot_to_keep/u_f_eigen_mode_{fine_eigenpairs[j][0]}.png")
				break

	####################################################################################################################

	# Calculate difference between LOD and fine solution and save plots for the three chosen modes
	print("L2 error between LOD and fine solution:")
	for i in range(len(lod_eigenpairs)):
		if lod_eigenpairs[i][2] is not None:
			diff1 = np.abs(lod_eigenpairs[i][2] - fine_eigenpairs[i][2])
			diff2 = np.abs(lod_eigenpairs[i][2] + fine_eigenpairs[i][2])  # it's possible that the modes are opposite
			norm1 = np.linalg.norm(diff1)
			norm2 = np.linalg.norm(diff2)
			if norm1 < norm2:
				diff = diff1
				norm = norm1
			else:
				diff = diff2
				norm = norm2

			screenshot(msh_f, diff, f"plot_to_keep/fine-LOD_diff_mode_{lod_eigenpairs[i][0]}.png")

			print(f" Mode {lod_eigenpairs[i][0]}: {norm:.4f}")


if __name__ == "__main__":
	main(num_mesh_refines=2, show_plots=False)
