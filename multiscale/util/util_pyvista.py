import pyvista as pv

from dolfinx import plot


def plot_grid_points(mesh, data, data_name, file_name, show_plots, cmap="seismic", show_edges=True):
	grid = pv.UnstructuredGrid(*plot.vtk_mesh(mesh, mesh.topology.dim))
	grid.point_data[data_name] = data
	grid.set_active_scalars(data_name)

	plotter = pv.Plotter(window_size=[1000, 1000], off_screen=not show_plots)
	plotter.show_axes()
	plotter.show_grid()
	plotter.add_mesh(grid, show_edges=show_edges, scalars=data_name, cmap=cmap)
	plotter.view_xy()

	if not show_plots:
		plotter.screenshot(file_name)
	else:
		plotter.show()
	plotter.close()


def plot_grid_cells(mesh, data, data_name, file_name, show_plots, cmap="seismic", show_edges=True):
	grid = pv.UnstructuredGrid(*plot.vtk_mesh(mesh, mesh.topology.dim))
	grid.cell_data[data_name] = data
	grid.set_active_scalars(data_name)

	plotter = pv.Plotter(window_size=[1000, 1000], off_screen=not show_plots)
	plotter.show_axes()
	plotter.show_grid()
	plotter.add_mesh(grid, show_edges=show_edges, scalars=data_name, cmap=cmap)
	plotter.view_xy()

	if not show_plots:
		plotter.screenshot(file_name)
	else:
		plotter.show()
	plotter.close()


def screenshot(mesh, f, filename, cmap="seismic"):
	plotter = pv.Plotter(window_size=[1000, 1000], off_screen=True)
	plotter.show_axes()
	plotter.show_grid()
	plotter.add_mesh(pv.UnstructuredGrid(*plot.vtk_mesh(mesh, mesh.topology.dim)), show_edges=False, scalars=f,
					 cmap=cmap)
	plotter.view_xy()
	plotter.screenshot(filename)
	plotter.close()
