import numpy as np
import scipy.sparse.linalg as spl


""" This file stores the various sparse linear system solvers you can use for FDFD """

# ----------------------------------------------------------------------
# Portable, high-performance solver selection
#  - x86_64 (Intel/AMD): prefer PARDISO via pypardiso or legacy pyMKL
#  - Apple Silicon / other arch: use SciPy; prefer UMFPACK if present
#  - Manual override: CEVICHE_SOLVER={pardiso,scipy}
# ----------------------------------------------------------------------
import os
import platform

_ARCH = platform.machine().lower()
_IS_X86_64 = _ARCH in ("x86_64", "amd64")
_ENV_SOLVER = os.getenv("CEVICHE_SOLVER", "").lower()

HAVE_PYPARDISO = False
HAVE_PYMKL = False
HAVE_UMFPACK = False

if _IS_X86_64:
    try:
        from pypardiso import spsolve as _pardiso_spsolve
        HAVE_PYPARDISO = True
    except Exception:
        pass
    try:
        from pyMKL import pardisoSolver as _pyMKL_pardisoSolver
        HAVE_PYMKL = True
    except Exception:
        pass

try:
    import scikits.umfpack as _umf
    HAVE_UMFPACK = True
except Exception:
    HAVE_UMFPACK = False

# Backward compatible flag (used later) and default policy
HAS_MKL = _IS_X86_64 and (HAVE_PYPARDISO or HAVE_PYMKL)
if _ENV_SOLVER in ("pardiso", "pypardiso"):
    HAS_MKL = HAS_MKL and _IS_X86_64
elif _ENV_SOLVER == "scipy":
    HAS_MKL = False

# default iterative method to use
# for reference on the methods available, see:  https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
DEFAULT_ITERATIVE_METHOD = 'bicg'

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    'bicg': spl.bicg,
    'bicgstab': spl.bicgstab,
    'cg': spl.cg,
    'cgs': spl.cgs,
    'gmres': spl.gmres,
    'lgmres': spl.lgmres,
    'qmr': spl.qmr,
    'gcrotmk': spl.gcrotmk
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """

def solve_linear(A, b, iterative_method=False):
    """Master function to call direct or iterative solvers.

    Args:
        A: sparse matrix (CSR/CSC preferred).
        b: right-hand side vector/array.
        iterative_method: False for direct; None for default iterative;
            or a string key in ITERATIVE_METHODS.
    """
    if iterative_method is False:
        return _solve_direct(A, b)
    if iterative_method is None:
        return _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD)
    return _solve_iterative(A, b, iterative_method=iterative_method)

def _solve_direct(A, b):
    """Direct solver.

    Policy:
        - On x86_64, prefer PARDISO if available.
        - Otherwise use SciPy spsolve.
        - Ensure CSC format for best performance (UMFPACK/SuperLU).
        - Preserve complex dtype and 1-D RHS.
    """
    import scipy.sparse as sp

    b = np.asarray(b)
    if b.ndim > 1:
        b = b.reshape((-1,))
    # Allow real or complex; cast only if needed by solver path
    if np.iscomplexobj(A) or np.iscomplexobj(b):
        b = b.astype(np.complex128, copy=False)

    A_csc = A if sp.isspmatrix_csc(A) else A.tocsc()

    if HAS_MKL:
        if HAVE_PYPARDISO:
            # pypardiso: spsolve-like API
            return _pardiso_spsolve(A_csc, b)
        if HAVE_PYMKL:
            # 13 = complex unsymmetric (SC-PML usually non-Hermitian)
            mtype = 13 if np.iscomplexobj(A_csc.data) else 11
            ps = _pyMKL_pardisoSolver(A_csc, mtype=mtype)
            ps.factor()
            x = ps.solve(b)
            ps.clear()
            return x

    # SciPy fallback; prefer UMFPACK if available
    try:
        return spl.spsolve(A_csc, b, use_umfpack=HAVE_UMFPACK)
    except TypeError:
        # Older SciPy without use_umfpack kw
        return spl.spsolve(A_csc, b)

def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD):
    """ Iterative solver """

    # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(iterative_method, ITERATIVE_METHODS))

    # call the solver using scipy's API
    x, info = solver_fn(A, b, atol=ATOL)
    return x

def _solve_cuda(A, b, **kwargs):
    """ You could put some other solver here if you're feeling adventurous """
    raise NotImplementedError("Please implement something fast and exciting here!")


""" ============================ SPEED TESTS ============================= """

# to run speed tests use `python -W ignore ceviche/solvers.py` to suppress warnings

if __name__ == '__main__':

    from scipy.sparse import csr_matrix
    from scipy.sparse import random as random_sp
    from numpy.random import random as random
    import numpy as np
    from time import time
    import ceviche

    N = 200       # dimension of the x, and b vectors
    density = 0.3  # sparsity of the dense matrix
    A = csr_matrix(random_sp(N, N, density=density))
    b = np.random.random((N, 1)) - 0.5

    print('\nWITH RANDOM MATRICES:\n')
    print('\tfor N = {} and density = {}\n'.format(N, density))

    # DIRECT SOLVE
    t0 = time()
    x = _solve_direct(A, b)
    t1 = time()
    print('\tdirect solver:\n\t\ttook {} seconds\n'.format(t1 - t0))

    # ITERATIVE SOLVES

    for iterative_method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, iterative_method=iterative_method)
        t1 = time()
        print('\titerative solver ({}):\n\t\ttook {} seconds'.format(iterative_method, t1 - t0))

    print('\n')

    print('WITH FDFD MATRICES:\n')

    m, n = 200, 100
    print('\tfor dimensions = {}\n'.format((m, n)))
    eps_r = np.random.random((m, n)) + 1
    b = np.random.random((m * n, )) - 0.5

    import sys
    sys.path.append('../ceviche')
    from ceviche.fdfd import fdfd_ez as fdfd
    from ceviche.constants import *

    npml = 10
    dl = 2e-8
    lambda0 = 1550e-9
    omega0 = 2 * np.pi * C_0 / lambda0

    F = fdfd(omega0, dl, eps_r, [10, 0])
    entries_a, indices_a = F.make_A(eps_r.flatten())
    A = ceviche.primitives.make_sparse(entries_a, indices_a, m*n)

    # DIRECT SOLVE
    t0 = time()
    x = _solve_direct(A, b)
    t1 = time()
    print('\tdirect solver:\n\t\ttook {} seconds\n'.format(t1 - t0))

    # ITERATIVE SOLVES

    for iterative_method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, iterative_method=iterative_method)
        t1 = time()
        print('\titerative solver ({}):\n\t\ttook {} seconds'.format(iterative_method, t1 - t0))
