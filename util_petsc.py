from petsc4py import PETSc
from scipy.sparse import csr_matrix

def csr_to_petsc(csr: csr_matrix) -> PETSc.Mat:
	"""Convert a CSR matrix to a PETSc matrix."""
	mat = PETSc.Mat().createDense(csr.shape)
	mat.setValuesCSR(csr.indptr, csr.indices, csr.data)
	mat.assemble()
	return mat