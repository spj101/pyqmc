import numpy as np
import pyqmc
import math

#
# Function Generator From dmaitre
#
def makeFn1(ndiff, nfix):
    u1 = np.random.random()
    a = np.random.random((ndiff+nfix,))
    def f(x):
        return np.cos(u1+ np.dot(x,a))
    arest = a[ndiff:]
    adiff = a[:ndiff]
    norm = 2**ndiff*np.prod(np.sin(adiff/2))/np.prod(adiff)
    arg = u1 + sum(adiff)/2
    def F(x):
        return norm*np.cos(arg+np.dot(x,arest))
    return f,F


def makeFn2(ndiff, nfix):
    u = np.random.random((ndiff+nfix,))
    a = np.random.random((ndiff+nfix,))
    ainv = 1.0/a**2
    def f(x):
        diff = x-u
        return np.prod(ainv+diff**2, axis=0)
    arest = a[ndiff:]
    adiff = a[:ndiff]
    urest = u[ndiff:]
    udiff = u[:ndiff]
    arestinv = 1.0/arest**2
    integrated = 1/3.0 + 1/adiff**2 - udiff + udiff**2
    prefactor = np.prod(integrated)
    def F(x):
        diff = x-urest
        return prefactor * np.prod(arestinv+diff**2, axis=0)
    return f,F

#
# Plotting script
#
def generate_plot(f,f_k,F):
    ns = 32
    for i in range(2,5+1):
        res = pyqmc.qmc_integrate(f, int(10**i/ns), ns, nx)
        resk = pyqmc.qmc_integrate(f_k33, int(10**i/ns), ns, nx)
        print('==','n = 10**',i,'dim =',nx,'==')
        print(res)
        print(resk)
        print(F([]))
        print( abs(math.log10(abs(1.-res[0]/F([])))) )
        print( abs(math.log10(abs(1.-resk[0]/F([])))) )


if __name__ == '__main__':
    
    nx = 2

    # Function 1
    f, F = makeFn1(nx, 0)
    def f_k11(x):
        # function with Korobov a=1,b=1 applied
        wgt = 1.
        for i in range(len(x)):
            wgt *= 6. * (1 - x[i]) * x[i]
            x[i] = x[i] ** 2 * (3. - 2. * x[i] )
        return wgt * f(x)

    def f_k33(x):
        # function with Korobov a=3,b=3 applied
        wgt = 1.
        for i in range(len(x)):
            wgt *= 140. * (1 - x[i]) ** 3 * x[i] ** 3
            x[i] = x[i] ** 4 * (35. + x[i] * (-84. + (70. - 20. * x[i]) * x[i]))
        return wgt * f(x)
        
    generate_plot(f,f_k33,F)
    
    # Function 2
    f, F = makeFn2(nx, 0)
    def f_k11(x):
        # function with Korobov a=1,b=1 applied
        wgt = 1.
        for i in range(len(x)):
            wgt *= 6. * (1 - x[i]) * x[i]
            x[i] = x[i] ** 2 * (3. - 2. * x[i] )
        return wgt * f(x)

    def f_k33(x):
        # function with Korobov a=3,b=3 applied
        wgt = 1.
        for i in range(len(x)):
            wgt *= 140. * (1 - x[i]) ** 3 * x[i] ** 3
            x[i] = x[i] ** 4 * (35. + x[i] * (-84. + (70. - 20. * x[i]) * x[i]))
        return wgt * f(x)
        
    generate_plot(f,f_k33,F)

    
