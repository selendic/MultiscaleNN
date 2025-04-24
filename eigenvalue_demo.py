import sys

from mpi4py import MPI

import numpy as np
#from analytical_modes import verify_mode

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, fem, io, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import (CellType, create_rectangle, exterior_facet_indices,
                          locate_entities)

try:
    import pyvista
    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

try:
    from slepc4py import SLEPc
except ModuleNotFoundError:
    print("slepc4py is required for this demo")
    sys.exit(0)



w = 1
h = 0.45 * w
d = 0.5 * h
nx = 300
ny = int(0.4 * nx)

msh = create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [w, h]]), np.array([nx, ny]), CellType.quadrilateral)
msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)



eps_v = 1
eps_d = 2.45


def Omega_d(x):
    return x[1] <= d


def Omega_v(x):
    return x[1] >= d


D = fem.functionspace(msh, ("DQ", 0))
eps = fem.Function(D)

cells_v = locate_entities(msh, msh.topology.dim, Omega_v)
cells_d = locate_entities(msh, msh.topology.dim, Omega_d)

eps.x.array[cells_d] = np.full_like(cells_d, eps_d, dtype=default_scalar_type)
eps.x.array[cells_v] = np.full_like(cells_v, eps_v, dtype=default_scalar_type)



degree = 1
RTCE = element("RTCE", msh.basix_cell(), degree)
Q = element("Lagrange", msh.basix_cell(), degree)
V = fem.functionspace(msh, mixed_element([RTCE, Q]))



lmbd0 = h / 0.2
k0 = 2 * np.pi / lmbd0

et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - (k0**2) * eps * ufl.inner(et, vt)) * ufl.dx
b_tt = ufl.inner(et, vt) * ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - (k0**2) * eps * ufl.inner(ez, vz)) * ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)



bc_facets = exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)



A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc])
B.assemble()

eps = SLEPc.EPS().create(msh.comm)

eps.setOperators(A, B)

eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

tol = 1e-9
eps.setTolerances(tol=tol)

eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

# Get ST context from eps
st = eps.getST()

# Set shift-and-invert transformation
st.setType(SLEPc.ST.Type.SINVERT)

eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

eps.setTarget(-(0.5 * k0)**2)

eps.setDimensions(nev=1)



eps.solve()
eps.view()
eps.errorView()



# Save the kz
vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())]

# Sort kz by real part
vals.sort(key=lambda x: x[1].real)

eh = fem.Function(V)

kz_list = []

for i, kz in vals:
    # Save eigenvector in eh
    eps.getEigenpair(i, eh.vector)

    # Compute error for i-th eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)

    # Verify, save and visualize solution
    if error < tol and np.isclose(kz.imag, 0, atol=tol):
        kz_list.append(kz)

        # Verify if kz is consistent with the analytical equations
#        assert verify_mode(kz, w, h, d, lmbd0, eps_d, eps_v, threshold=1e-4)

        print(f"eigenvalue: {-kz**2}")
        print(f"kz: {kz}")
        print(f"kz/k0: {kz/k0}")

        eh.x.scatter_forward()

        eth, ezh = eh.split()
        eth = eh.sub(0).collapse()
        ez = eh.sub(1).collapse()

        # Transform eth, ezh into Et and Ez
        eth.x.array[:] = eth.x.array[:] / kz
        ezh.x.array[:] = ezh.x.array[:] * 1j

        gdim = msh.geometry.dim
        V_dg = fem.functionspace(msh, ("DQ", degree, (gdim,)))
        Et_dg = fem.Function(V_dg)
        Et_dg.interpolate(eth)

        # Save solutions
        with io.VTXWriter(msh.comm, f"sols/Et_{i}.bp", Et_dg) as f:
            f.write(0.0)

        with io.VTXWriter(msh.comm, f"sols/Ez_{i}.bp", ezh) as f:
            f.write(0.0)

        # Visualize solutions with Pyvista
        if have_pyvista:
            V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            Et_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
            Et_values[:, : msh.topology.dim] = Et_dg.x.array.reshape(V_x.shape[0], msh.topology.dim).real

            V_grid.point_data["u"] = Et_values

            plotter = pyvista.Plotter()
            plotter.add_mesh(V_grid.copy(), show_edges=False)
            plotter.view_xy()
            plotter.link_views()
            if not pyvista.OFF_SCREEN:
                plotter.show()
            else:
                pyvista.start_xvfb()
                plotter.screenshot("Et.png", window_size=[400, 400])

        if have_pyvista:
            V_lagr, lagr_dofs = V.sub(1).collapse()
            V_cells, V_types, V_x = plot.vtk_mesh(V_lagr)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            V_grid.point_data["u"] = ezh.x.array.real[lagr_dofs]
            plotter = pyvista.Plotter()
            plotter.add_mesh(V_grid.copy(), show_edges=False)
            plotter.view_xy()
            plotter.link_views()
            if not pyvista.OFF_SCREEN:
                plotter.show()
            else:
                pyvista.start_xvfb()
                plotter.screenshot("Ez.png", window_size=[400, 400])
