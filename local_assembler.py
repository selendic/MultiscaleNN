import cffi

import numpy as np
from dolfinx import fem
from petsc4py.lib.PETSc import ScalarType

# Local assembler for a bilinear form
# https://fenicsproject.discourse.group/t/how-to-enumerate-the-cells-of-a-mesh-with-dolfinx-v0-6-0/11661/7
_type_to_offset_index = {fem.IntegralType.cell: 0, fem.IntegralType.exterior_facet: 1}


class LocalAssembler:

	def __init__(self, form, integral_type: fem.IntegralType = fem.IntegralType.cell):
		self.consts = None
		self.coeffs = None
		self.form = fem.form(form)
		self.integral_type = integral_type
		self.update_coefficients()
		self.update_constants()

		subdomain_ids = self.form._cpp_object.integral_ids(integral_type)
		assert (len(subdomain_ids) == 1)
		assert (subdomain_ids[0] == -1)
		is_complex = np.issubdtype(ScalarType, np.complexfloating)
		nptype = "complex128" if is_complex else "float64"
		o_s = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type]]
		o_e = self.form.ufcx_form.form_integral_offsets[_type_to_offset_index[integral_type] + 1]
		assert o_e - o_s == 1

		self.kernel = getattr(self.form.ufcx_form.form_integrals[o_s], f"tabulate_tensor_{nptype}")
		self.active_cells = self.form._cpp_object.domains(integral_type, -1)
		assert len(self.form.function_spaces) == 2
		self.local_shape = [0, 0]
		for i, V in enumerate(self.form.function_spaces):
			self.local_shape[i] = V.dofmap.dof_layout.block_size * V.dofmap.dof_layout.num_dofs

		e0 = self.form.function_spaces[0].element
		e1 = self.form.function_spaces[1].element
		needs_transformation_data = e0.needs_dof_transformations or e1.needs_dof_transformations or \
									self.form._cpp_object.needs_facet_permutations
		if needs_transformation_data:
			raise NotImplementedError("Dof transformations not implemented")

		self.ffi = cffi.FFI()
		V = self.form.function_spaces[0]
		self.x_dofs = V.mesh.geometry.dofmap

	def update_coefficients(self):
		self.coeffs = fem.assemble.pack_coefficients(self.form._cpp_object)[(self.integral_type, -1)]

	def update_constants(self):
		self.consts = fem.assemble.pack_constants(self.form._cpp_object)

	def update(self):
		self.update_coefficients()
		self.update_constants()

	def assemble_matrix(self, i: int, local_index: int = 0):

		x = self.form.function_spaces[0].mesh.geometry.x
		x_dofs = self.x_dofs[i]
		geometry = np.zeros((len(x_dofs), 3), dtype=np.float64)
		geometry[:, :] = x[x_dofs]

		A_local = np.zeros((self.local_shape[0], self.local_shape[0]), dtype=ScalarType)
		facet_index = np.array([local_index], dtype=np.intc)
		facet_perm = np.zeros(0, dtype=np.uint8)
		if self.coeffs.shape == (0, 0):
			coeffs = np.zeros(0, dtype=ScalarType)
		else:
			coeffs = self.coeffs[i, :]
		ffi_fb = self.ffi.from_buffer
		self.kernel(ffi_fb(A_local), ffi_fb(coeffs), ffi_fb(self.consts), ffi_fb(geometry),
					ffi_fb(facet_index), ffi_fb(facet_perm))
		return A_local
