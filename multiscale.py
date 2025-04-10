from typing import Tuple

import dolfinx
import numpy as np
from dolfinx import mesh, fem
from scipy.linalg import inv
from scipy.sparse import csr_matrix, lil_matrix

import gmesh
from local_assembler import LocalAssembler


kappa_1 = lambda x: 3  # kappa in large cells
kappa_2 = lambda x: 50  # kappa in small cells

kappa_x = 0.6875 * gmesh.d
kappa_y = 0.6875 * gmesh.d
kappa_a2 = np.power(10.0, -4)

# V
gamma = 2e4
nu = 20.0


def create_kappa(
		msh: mesh.Mesh,
		ct: mesh.MeshTags,
		marker_1: int,
		marker_2: int
) -> Tuple[fem.Function, fem.Function]:
	# Create a discontinuous function for tagging the cells
	FS_DG = fem.functionspace(msh, ("DG", 0))
	q = fem.Function(FS_DG)
	cells_1 = ct.find(marker_1)
	q.x.array[cells_1] = np.full_like(cells_1, marker_1, dtype=dolfinx.default_scalar_type)
	cells_2 = ct.find(marker_2)
	q.x.array[cells_2] = np.full_like(cells_2, marker_2, dtype=dolfinx.default_scalar_type)

	# Out of q, we can create an arbitrary kappa:
	# We can now create our discontinuous function for tagging subdomains:
	FS_CG = fem.functionspace(msh, ("CG", 1))
	fx, fy, fz = fem.Function(FS_CG), fem.Function(FS_CG), fem.Function(FS_CG)
	fq, kappa = fem.Function(FS_CG), fem.Function(FS_CG)
	fx.interpolate(lambda x: x[0])
	fy.interpolate(lambda x: x[1])
	fz.interpolate(lambda x: x[2])
	fq.interpolate(q)
	for i in range(len(kappa.x.array)):
		x = [fx.x.array[i], fy.x.array[i], fz.x.array[i]]
		if fq.x.array[i] == marker_2:
			kappa.x.array[i] = kappa_2(x)
		else:
			kappa.x.array[i] = kappa_1(x)

	return q, kappa


def create_point_source(
		msh: mesh.Mesh,
		x0: float,
		y0: float,
		a2: float
) -> fem.Function:
	FS_ps = fem.functionspace(msh, ("CG", 1))
	f = fem.Function(FS_ps)
	dirac = lambda x: 1.0 / np.sqrt(a2 * np.pi) * np.exp(
		-(np.power(x[0] - x0, 2) + np.power(x[1] - y0, 2)) / a2
	)
	f.interpolate(dirac)
	return f


def create_v(FS: fem.FunctionSpace) -> fem.Function:
	# Constants

	# Define a Python function for the potential
	def v_np(x):
		# x[0], x[1] are arrays of coordinates
		return np.ceil(gamma * np.cos(np.pi * nu * (x[0] + 0.1)) * np.cos(np.pi * nu * x[1]))

	# Interpolate into a FEniCSx function
	v_expr = fem.Function(FS)
	v_expr.interpolate(v_np)

	return v_expr


def compute_correction_operator(
		msh_c: mesh.Mesh,
		ct_c: mesh.MeshTags,
		parent_cells: np.ndarray,
		FS_c: fem.FunctionSpace,
		FS_f: fem.FunctionSpace,
		coarse_boundary_dofs: np.ndarray,
		fine_boundary_dofs: np.ndarray,
		A_h: csr_matrix,
		P_h: csr_matrix,
		C_h: csr_matrix,
		fine_stiffness_assembler: LocalAssembler
) -> lil_matrix:
	num_dofs_c = FS_c.dofmap.index_map.size_local * FS_c.dofmap.index_map_bs  # N_H
	num_dofs_f = FS_f.dofmap.index_map.size_local * FS_f.dofmap.index_map_bs  # N_h

	Q_h = lil_matrix((num_dofs_c, num_dofs_f))

	# For each coarse cell K_l
	for l in ct_c.indices:
		# Create local patch U_l consisting of K_l with one layer of neighboring cells (k = 1, for now)
		# https://docs.fenicsproject.org/dolfinx/main/cpp/mesh.html#_CPPv4N7dolfinx4mesh25compute_incident_entitiesERK8TopologyNSt4spanIKNSt7int32_tEEEii
		incident_facets = mesh.compute_incident_entities(msh_c.topology, l, 2, 1)
		incident_vertices = mesh.compute_incident_entities(msh_c.topology, l, 2, 0)
		coarse_patch_1 = mesh.compute_incident_entities(msh_c.topology, incident_facets, 1, 2)
		coarse_patch_2 = mesh.compute_incident_entities(msh_c.topology, incident_vertices, 0, 2)
		coarse_patch = np.unique(np.concatenate((coarse_patch_1, coarse_patch_2)))

		# Find coarse dofs on patch
		coarse_dofs_local = fem.locate_dofs_topological(FS_c, 2, coarse_patch)  # Z_l[i] = coarse_dofs_local[i]
		coarse_dofs_local = np.setdiff1d(coarse_dofs_local, coarse_boundary_dofs, assume_unique=True)
		num_coarse_dofs_local = coarse_dofs_local.size  # N_H_l

		# Create restriction matrix R_H_l (N_H_l x N_H)
		R_H_l = csr_matrix(
			(
				np.ones_like(coarse_dofs_local),
				(
					np.arange(num_coarse_dofs_local),
					coarse_dofs_local
				),
			),
			shape=(num_coarse_dofs_local, num_dofs_c)
		)
		#        print("R_H_l:")
		#        print(R_H_l)

		# Find fine cells on patch
		fine_patch = np.where(np.isin(parent_cells, coarse_patch))[0]

		# Find fine dofs on patch
		fine_dofs_local = fem.locate_dofs_topological(FS_f, 2, fine_patch)  # z_l[i] = fine_dofs_local[i]
		fine_dofs_local = np.setdiff1d(fine_dofs_local, fine_boundary_dofs, assume_unique=True)
		num_fine_dofs_local = fine_dofs_local.size  # N_h_l

		# Create restriction matrix R_h_l (N_h_l x N_h)
		R_h_l = csr_matrix(
			(
				np.ones_like(fine_dofs_local),
				(
					np.arange(num_fine_dofs_local),
					fine_dofs_local
				),
			),
			shape=(num_fine_dofs_local, num_dofs_f)
		)
		assert R_h_l.shape == (num_fine_dofs_local, num_dofs_f)

		#        print("R_h_l:")
		#        print(R_h_l)

		# Create local coarse-node-to-coarse-element restriction matrix T_H_l (c_d x N_H)
		l_global_dofs = fem.locate_dofs_topological(FS_c, 2, l)  # p[i] = l_dofs[i]
		assert l_global_dofs.size == msh_c.topology.cell_types[0].value
		T_H_l = csr_matrix(
			(
				np.ones_like(l_global_dofs),
				(
					np.arange(l_global_dofs.size),
					l_global_dofs
				),
			),
			shape=(l_global_dofs.size, num_dofs_c)
		)
		assert T_H_l.shape == (l_global_dofs.size, num_dofs_c)

		#        print("T_H_l:")
		#        print(T_H_l)

		# Calculate local stiffness matrix and constraints matrix
		A_l = R_h_l @ A_h @ R_h_l.transpose()
		C_l = R_H_l @ C_h @ R_h_l.transpose()

		# In order to create local load vector matrix,
		# we need the contributions of local stiffness matrices on fine cells
		sigma_A_sigmaT_l = lil_matrix((num_dofs_f, num_dofs_f))
		# Find fine cells only on coarse cell l
		fine_cells_on_l = np.where(parent_cells == l)[0]
		for t in fine_cells_on_l:
			# Assemble local stiffness matrix A_t
			A_t = fine_stiffness_assembler.assemble_matrix(t)
			A_t = csr_matrix(A_t)

			# Find global fine dofs on fine cell t
			fine_dofs_global_t = fem.locate_dofs_topological(FS_f, 2, t)
			# Create local-to-global-mapping sigma_t
			sigma_t = csr_matrix(
				(
					np.ones_like(fine_dofs_global_t),
					(
						np.arange(fine_dofs_global_t.size),
						fine_dofs_global_t
					),
				),
				shape=(fine_dofs_global_t.size, num_dofs_f)
			)

			# Add the contribution
			sigma_A_sigmaT_l += sigma_t.transpose() @ A_t @ sigma_t

		# Create local load vector matrix
		r_l = - (T_H_l @ P_h @ sigma_A_sigmaT_l @ R_h_l.transpose())

		# Compute the inverse of the local stiffness matrix
		A_l_inv = csr_matrix(inv(A_l.todense()))

		# Precomputations related to the operator
		Y_l = A_l_inv @ C_l.transpose()

		# Compute inverse Schur complement
		S_l_inv = csr_matrix(inv((C_l @ Y_l).todense()))

		# Compute correction for each coarse space function with support on K_l
		w_l = lil_matrix((l_global_dofs.size, num_fine_dofs_local))
		for i in range(l_global_dofs.size):
			q_i = A_l_inv @ r_l[i].transpose()
			lambda_i = S_l_inv @ (C_l @ q_i)
			w_l_i = q_i - Y_l @ lambda_i
			w_l[i] = w_l_i.transpose()

		# Update the corrector matrix
		Q_h += T_H_l.transpose() @ w_l @ R_h_l

	return Q_h
