from typing import Tuple

import basix
import dolfinx
import meshio
import numpy as np
from dolfinx import geometry, mesh, fem
from dolfinx.cpp.mesh import to_string
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import csr_matrix


def csr_to_petsc(csr: csr_matrix) -> PETSc.Mat:
	"""Convert a CSR matrix to a PETSc matrix."""
	mat = PETSc.Mat().createDense(csr.shape)
	mat.setValuesCSR(csr.indptr, csr.indices, csr.data)
	mat.assemble()
	return mat


def interpolation_matrix_non_matching_meshes(V_1, V_0):  # Function spaces from non-matching meshes
	msh_0 = V_0.mesh
	x_0 = V_0.tabulate_dof_coordinates()
	x_1 = V_1.tabulate_dof_coordinates()

	bb_tree = geometry.bb_tree(msh_0, msh_0.topology.dim)
	cell_candidates = geometry.compute_collisions_points(bb_tree, x_1)
	cells = []
	points_on_proc = []
	index_points = []
	colliding_cells = geometry.compute_colliding_cells(msh_0, cell_candidates, x_1)

	for i, point in enumerate(x_1):
		if len(colliding_cells.links(i)) > 0:
			points_on_proc.append(point)
			cells.append(colliding_cells.links(i)[0])
			index_points.append(i)

	index_points_ = np.array(index_points)
	points_on_proc_ = np.array(points_on_proc, dtype=np.float64)
	cells_ = np.array(cells)

	ct = to_string(msh_0.topology.cell_types[0])
	element = basix.create_element(basix.finite_element.string_to_family(
		"Lagrange", ct), basix.cell.string_to_type(ct), V_0.ufl_element().degree(), basix.LagrangeVariant.equispaced)

	x_ref = np.zeros((len(cells_), 2))

	for i in range(0, len(cells_)):
		geom_dofs = msh_0.geometry.dofmap[cells_[i]]
		x_ref[i, :] = msh_0.geometry.cmaps[0].pull_back([points_on_proc_[i, :]], msh_0.geometry.x[geom_dofs])

	basis_matrix = element.tabulate(0, x_ref)[0, :, :, 0]

	cell_dofs = np.zeros((len(x_1), len(basis_matrix[0, :])))
	basis_matrix_full = np.zeros((len(x_1), len(basis_matrix[0, :])))

	for nn in range(0, len(cells_)):
		cell_dofs[index_points_[nn], :] = V_0.dofmap.cell_dofs(cells_[nn])
		basis_matrix_full[index_points_[nn], :] = basis_matrix[nn, :]

	cell_dofs_ = cell_dofs.astype(int)  # REDUCE HERE

	I = np.zeros((len(x_1), len(x_0)))

	for i in range(0, len(x_1)):
		for j in range(0, len(basis_matrix[0, :])):
			I[i, cell_dofs_[i, j]] = basis_matrix_full[i, j]

	return I


# A convenience function for extracting data for a single cell type, and creating a new meshio mesh,
# including physical markers for the given type.
def create_mesh(in_mesh, cell_type, prune_z=False) -> meshio.Mesh:
	cells = in_mesh.get_cells_type(cell_type)
	cell_data = in_mesh.get_cell_data("gmsh:physical", cell_type)
	points = in_mesh.points[:, :2] if prune_z else in_mesh.points
	out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},
						   cell_data={"name_to_read": [cell_data.astype(np.int32)]})
	return out_mesh


# We have now written the mesh and the cell markers to one file, and the facet markers in a separate file.
# We can now read this data in DOLFINx using XDMFFile.read_mesh and XDMFFile.read_meshtags.
# The dolfinx.MeshTags stores the index of the entity, along with the value of the marker in two one dimensional arrays.
def read_mesh(fine: bool, type: str) -> Tuple[mesh.Mesh, mesh.MeshTags, mesh.MeshTags]:
	if type not in ["triangle", "quad"]:
		raise ValueError("type must be either 'triangle' or 'quad'")
	suffix = "f" if fine else "c"
	# Read in mesh
	msh = meshio.read("data/mesh_" + suffix + ".msh")

	# Create and save one file for the mesh, and one file for the facets
	cell_mesh = create_mesh(msh, type, prune_z=True)
	line_mesh = create_mesh(msh, "line", prune_z=True)
	meshio.write("data/mesh_" + suffix + ".xdmf", cell_mesh)
	meshio.write("data/mt_" + suffix + ".xdmf", line_mesh)
	MPI.COMM_WORLD.barrier()

	# We read the mesh in parallel
	with XDMFFile(MPI.COMM_WORLD, "data/mesh_" + suffix + ".xdmf", "r") as xdmf:
		mesh = xdmf.read_mesh(name="Grid")  # Mesh
		ct = xdmf.read_meshtags(mesh, name="Grid")  # Cell tags
	# Create connectivity cell -> facet
	mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
	with XDMFFile(MPI.COMM_WORLD, "data/mt_" + suffix + ".xdmf", "r") as xdmf:
		ft = xdmf.read_meshtags(mesh, name="Grid")  # Facet tags

	# Create connectivity cell <-> point
	mesh.topology.create_connectivity(0, mesh.topology.dim)

	# print(mesh/ct/ft.indices)
	# print(mesh/ct/ft.values)

	return mesh, ct, ft
