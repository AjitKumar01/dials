"""DIALS: diagonal-plus-low-rank label relations with spectral cross-fitting, for bundle completion."""
import os
import sys
import warnings

# One BLAS thread, set before numpy loads: the randomized SVDs (features, label factors) are bit-reproducible only at
# a fixed thread count, and Apple Accelerate cannot be limited after it starts.
if 'numpy' in sys.modules and os.environ.get('VECLIB_MAXIMUM_THREADS')!='1':
    warnings.warn('numpy was imported before dials without VECLIB_MAXIMUM_THREADS=1; SVD caches may not be bit-reproducible')
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key,'1')
