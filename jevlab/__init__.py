import os

# BLAS/OpenMP otherwise start one busy-waiting thread per core; the matrices here are too small to use them.
_threads = os.environ.get("JEV_THREADS") or str(min(8, os.cpu_count() or 8))
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _threads)
