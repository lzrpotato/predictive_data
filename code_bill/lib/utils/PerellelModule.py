import numpy as np
import scipy.sparse as sp
from joblib import Parallel, cpu_count, delayed


def _predict(estimator, X, method, start, stop):
    return getattr(estimator, method)(X[start:stop])


def parallel_predict(estimator, X, n_jobs=1, method='predict', batches_per_job=3):
    n_jobs = max(cpu_count() + 1 + n_jobs, 1)
    n_batches = batches_per_job * n_jobs
    n_samples = len(X)
    batch_size = int(np.ceil(n_samples / n_batches))
    parallel = Parallel(n_jobs=n_jobs)
    results = parallel(delayed(_predict)(estimator, X, method, i, i + batch_size)
                       for i in range(0, n_samples, batch_size))
    if sp.issparse(results[0]):
        return sp.vstack(results)
    return np.concatenate(results)
