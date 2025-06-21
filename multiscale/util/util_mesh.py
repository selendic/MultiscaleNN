from typing import Tuple

import basix
import dolfinx
import meshio
import numpy as np
from dolfinx import fem, geometry, mesh
from dolfinx.cpp.mesh import to_string
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.io import XDMFFile
from mpi4py import MPI

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "multiscale"))
import gmesh

kappa_1 = lambda x: 3  # kappa in large cells
kappa_2 = lambda x: 50  # kappa in small cells

kappa_x = 0.6875 * gmesh.d
kappa_y = 0.6875 * gmesh.d
kappa_a2 = np.power(10.0, -4)


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


def refine_mesh(msh_c: mesh.Mesh, ct_c: mesh.MeshTags, marker_facet_boundary: int):
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


def create_v(FS: fem.FunctionSpace, gamma: float, nu: float) -> fem.Function:
	# Constants

	# Define a Python function for the potential
	def v_np(x):
		# x[0], x[1] are arrays of coordinates
		return np.ceil(gamma * np.cos(np.pi * nu * (x[0] + 0.1)) * np.cos(np.pi * nu * x[1]))

	# Interpolate into a FEniCSx function
	v_expr = fem.Function(FS)
	v_expr.interpolate(v_np)

	return v_expr
