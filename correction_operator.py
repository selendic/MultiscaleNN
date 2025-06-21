import numpy as np
from dolfinx import mesh, fem
from scipy.linalg import inv
from scipy.sparse import csr_matrix, lil_matrix

from local_assembler import LocalAssembler


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
