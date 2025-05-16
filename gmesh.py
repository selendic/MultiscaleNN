import gmsh
import numpy as np

marker_cell_outer = 3
marker_cell_inner = 50
marker_facet_boundary = 1

n = 4  # number of subdomain per row/column
d = 1.0 / n  # size of square subdomain
lc_outer = 0.025 * 3
lc_inner = 0.025 * 3
lc_simple = 2e-2 * 2


def create_simple_gmsh(size: tuple[int, int]):

	w, h = size

	gmsh.initialize()

	p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc_simple)
	p2 = gmsh.model.occ.addPoint(w, 0.0, 0.0, lc_simple)
	p3 = gmsh.model.occ.addPoint(w, h, 0.0, lc_simple)
	p4 = gmsh.model.occ.addPoint(0.0, h, 0.0, lc_simple)

	l1 = gmsh.model.occ.addLine(p1, p2)
	l2 = gmsh.model.occ.addLine(p2, p3)
	l3 = gmsh.model.occ.addLine(p3, p4)
	l4 = gmsh.model.occ.addLine(p4, p1)

	cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
	ps = gmsh.model.occ.addPlaneSurface([cl])

	gmsh.model.occ.synchronize()

	# Tag the cells
	gmsh.model.addPhysicalGroup(2, [ps], marker_cell_outer)

	# Tag the facets
	facets = []
	for line in gmsh.model.getEntities(dim=1):
		com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
		if np.isclose(com[0], 0.0) or np.isclose(com[0], w) or np.isclose(com[1], 0.0) or np.isclose(com[1], h):
			facets.append(line[1])
	gmsh.model.addPhysicalGroup(1, facets, marker_facet_boundary)

	gmsh.model.mesh.generate(2)
	gmsh.write("data/mesh_c.msh")

	gmsh.finalize()


def create_gmsh(size: tuple[int, int], fine: bool = False):

	w, h = size

	gmsh.initialize()

	gmsh.option.setNumber("General.Verbosity", 2)
	outer_cell_tags = []
	inner_cell_tags = []
	ones, others = [], []
	boundary_tags = []
	#if proc == 0:
	# Create the outer domain
	p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc_outer)
	p2 = gmsh.model.occ.addPoint(w, 0.0, 0.0, lc_outer)
	p3 = gmsh.model.occ.addPoint(w, h, 0.0, lc_outer)
	p4 = gmsh.model.occ.addPoint(0.0, h, 0.0, lc_outer)

	l1 = gmsh.model.occ.addLine(p1, p2)
	l2 = gmsh.model.occ.addLine(p2, p3)
	l3 = gmsh.model.occ.addLine(p3, p4)
	l4 = gmsh.model.occ.addLine(p4, p1)

	cl_outer = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
	ps_outer = gmsh.model.occ.addPlaneSurface([cl_outer])
	outer_cell_tags.append(ps_outer)
	ones.append((2, ps_outer))

	# Create the inner rectangles
	for i in range(n):
		for j in range(n):
			# We create one large cell, and one small cell within it
			# tag1 = i*n+j
			# tag2 = i*n+j + n*n

			# Create inner rectangle
			# gmsh.model.occ.addRectangle(d*(i+0.5), d*(j+0.5), 0, d*3/8, d*3/8, tag=tag2)
			p1 = gmsh.model.occ.addPoint(d * (i + 0.500), d * (j + 0.500), 0, lc_inner)
			p2 = gmsh.model.occ.addPoint(d * (i + 0.875), d * (j + 0.500), 0, lc_inner)
			p3 = gmsh.model.occ.addPoint(d * (i + 0.875), d * (j + 0.875), 0, lc_inner)
			p4 = gmsh.model.occ.addPoint(d * (i + 0.500), d * (j + 0.875), 0, lc_inner)
			l1 = gmsh.model.occ.addLine(p1, p2)
			l2 = gmsh.model.occ.addLine(p2, p3)
			l3 = gmsh.model.occ.addLine(p3, p4)
			l4 = gmsh.model.occ.addLine(p4, p1)

			cl_inner = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
			ps_inner = gmsh.model.occ.addPlaneSurface([cl_inner])

			# Create outer rectangle
			# gmsh.model.occ.addRectangle(d*i, d*j, 0, d, d, tag=tag1)
			#p1 = gmsh.model.occ.addPoint(d * i, d * j, 0, lc_outer)
			#p2 = gmsh.model.occ.addPoint(d * (i + 1), d * j, 0, lc_outer)
			#p3 = gmsh.model.occ.addPoint(d * (i + 1), d * (j + 1), 0, lc_outer)
			#p4 = gmsh.model.occ.addPoint(d * i, d * (j + 1), 0, lc_outer)
			#l1 = gmsh.model.occ.addLine(p1, p2)
			#l2 = gmsh.model.occ.addLine(p2, p3)
			#l3 = gmsh.model.occ.addLine(p3, p4)
			#l4 = gmsh.model.occ.addLine(p4, p1)

			#cl_outer = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
			#ps_outer = gmsh.model.occ.addPlaneSurface([cl_outer, cl_inner])

			# We add the rectangles in the subdomain list
			#outer_cell_tags.append(ps_outer)
			inner_cell_tags.append(ps_inner)
			others.append((2, ps_inner))

			# We add the appropriate rectangles to appropriate list for fragmenting
			#if (i + j) % 2 == 0:
			#	ones.append((2, ps_outer))
			#	others.append((2, ps_inner))
			#else:
			#	ones.append((2, ps_inner))
			#	others.append((2, ps_outer))

	outer_cell_tags[0] = gmsh.model.occ.fragment(ones, others)[1][0][0][1]

	gmsh.model.occ.synchronize()

	# print(outer_tags)
	# print(inner_tags)
	gmsh.model.addPhysicalGroup(2, outer_cell_tags, marker_cell_outer)
	gmsh.model.addPhysicalGroup(2, inner_cell_tags, marker_cell_inner)

	# Tag the dirichlet boundary facets
	for line in gmsh.model.getEntities(dim=1):
		com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
		if np.isclose(com[0], 0.0) or np.isclose(com[0], w) or np.isclose(com[1], 0.0) or np.isclose(com[1], h):
			boundary_tags.append(line[1])
	gmsh.model.addPhysicalGroup(1, boundary_tags, marker_facet_boundary)

	gmsh.model.mesh.generate(2)
	if fine:
		gmsh.model.mesh.refine()
	gmsh.write("data/mesh_" + ("f" if fine else "c") + ".msh")

	gmsh.finalize()
