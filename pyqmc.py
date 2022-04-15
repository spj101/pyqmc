import math
import numpy as np
import gvecs


def qmc_r1sl_sample(n, d, shift=None, gvs=gvecs.cbcpt_dn1_100):
    """
    Generate a set of rank-1 shifted-lattice points

    :param n:
        Minimum number of points to generate (will be rounded up to the next largest generating vector)
    :param d:
        Dimension of points to generate
    :param shift:
        Shift of lattice (must have `dim` dimensions)
    :param gvs:
        The generating vectors of the Rank-1 Lattice
    :return:
        Array of lattice points
    """
    gv_n = list(filter(lambda i: i >= n, gvs.keys()))[0]  # get key of first gen vec >= n
    gv = np.array(gvs[gv_n][:d])
    if shift is None:
        shift = np.zeros(d)
    res = np.array([[math.modf(x)[0] for x in ((i * gv) % gv_n) / float(gv_n) + shift] for i in range(gv_n)])
    return res


def qmc_integrate(func, n, s, d):
    """
    Integrate a function using rank-1 shifted lattices

    :param func:
        The integrand
    :param n:
        The minimum number of points to generate (will be rounded up to the next largest generating vector)
    :param s:
        Number of random shifts of the lattice to use
    :param d:
        (Input) Dimension of the integrand
    :return:
        Result of the integral and an estimate of the uncertainty in a tuple:
        ( result, error)
    """
    shift_results = []
    for _ in range(shifts):
        s = np.random.uniform(size=d) # generate shift
        x_lattice = qmc_r1sl_sample(lattice_size, d, s) # generate lattice
        shift_results.append(np.mean([func(x) for x in x_lattice ])) # each shift is avg of lattice
    return np.mean(shift_results), np.std(shift_results) # result is avg, std of shifts


if __name__ == '__main__':

    # Some test functions to see how this implementation works
    def test_func(x):
        return x[0]*x[1]*x[2]

    def test_func_k33(x):
        # test function but with Korobov a=3,b=3 applied
        wgt = 1.
        for i in range(len(x)):
            wgt *= 140. * (1 - x[i]) ** 3 * x[i] ** 3
            x[i] = x[i] ** 4 * (35. + x[i] * (-84. + (70. - 20. * x[i]) * x[i]))
        return wgt * test_func(x)

    # Input Parameters
    dim = 3 # Dimension of function
    shifts = 32 # Typically 32-64 (used to compute std)
    lattice_size = 1021 # Typically large
    np.random.seed(42) # DANGER: Ugly hack to make output reproducible!

    # Rank-1 Shifted Lattice Integration
    res_qmc = qmc_integrate(test_func, lattice_size, shifts, dim)
    print('qmc:', res_qmc[0], '+/-', res_qmc[1])

    # Rank-1 Shifted Lattice Integration (with periodisation)
    res_qmc = qmc_integrate(test_func_k33, lattice_size, shifts, dim)
    print('qmc,k33:', res_qmc[0], '+/-', res_qmc[1])

    # Monte Carlo on lattice_size * shifts points
    res_mc = [test_func(np.random.uniform(size=dim)) for _ in range(lattice_size*shifts)]
    res_mc = (np.mean(res_mc), np.std(res_mc))
    print('mc:', res_mc[0], '+/-', res_mc[1])

    # Monte Carlo on lattice_size * shifts points (with periodisation)
    res_mc = [test_func_k33(np.random.uniform(size=dim)) for _ in range(lattice_size*shifts)]
    res_mc = (np.mean(res_mc), np.std(res_mc))
    print('mc,k33:', res_mc[0], '+/-', res_mc[1])

