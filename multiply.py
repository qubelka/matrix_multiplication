import dask.array as da
from dask.diagnostics import ProgressBar, ResourceProfiler
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def multiply_matrices():
    """
    A function that creates three matrices A, B and C with random values 
    from the range [0.0, 1.0) of sizes:
    (1e6, 1e3)
    (1e3, 1e6)
    (1e6, 1)
    and calculates the product D = A * B * C. 

    To calculate the CDF of the A's values, A is first initialized as a 1D-array.
    First 1000 values are used for creating an ECDF plot. 
    After that the matrix A with dimensions (1e6, 1e3) is created from the array values.

    The function creates two plots: 1) ECDF-plot and 2) ResorceProfiler plot which shows CPU and memory usage.
    """

    # Dask built-in diagnostics utilities ProgressBar (program progress)
    # ResourceProfiler (CPU and memory usage) used to measure preformance.
    # ResourceProfiler's timeset is set to 300s = 5min
    pbar = ProgressBar()
    rprof = ResourceProfiler(dt=300)

    rprof.register()
    pbar.register()

    A = da.random.random(size=1e9, chunks=10000000)
    B = da.random.random(size=(1e3, 1e6), chunks=('auto', 10000))
    C = da.random.random(size=(1e6, 1), chunks=(10000, 'auto'))

    sample = A[:1000]
    sample.compute()
    ecdf = ECDF(sample)
    plt.plot(ecdf.x, ecdf.y)
    plt.savefig('ecdf.png')
    A = A.reshape(1e6, 1e3).rechunk(10000, 'auto')
    D = A.dot(B).dot(C)
    D.compute()

    rprof.visualize()

if __name__ == '__main__':
    multiply_matrices()
