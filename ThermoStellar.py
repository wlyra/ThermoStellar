import numpy as np
import pylab as plt
import astropy.constants as const
import sys

def get_grid_indices(nx,ng):
    mx = nx + 2*ng    
    i1=ng             # first computational point
    i2=mx-ng-1        # last computational point
    l1=i1
    l2=i2+1
    return mx,l1,l2,i1,i2

nx = 1000
ng = 3
mx,l1,l2,i1,i2 = get_grid_indices(nx,ng)
x0  = 1
xn  = 50
Lx = xn-x0
dx = Lx/(nx-1)

x = np.linspace(x0-3*dx,xn+3*dx,mx)


def der(f,x):
    n=len(x[l1:l2])
    dx = np.diff(x)[0]
    df = np.zeros(n)
    for i in range(n):
        j=i+i1
        df[i] =  -1./60*(f[j-3] - f[j+3])                  +3./20*(f[j-2] - f[j+2])                  -3./4 *(f[j-1] - f[j+1])        
    return df/dx

def der2(f,x):
    n=len(x[l1:l2])
    dx = np.diff(x)[0]
    d2f = np.zeros(n)    
    for i in range(n):
        j=i+i1
        d2f[i] =  1./90*(f[j-3] + f[j+3])                  -3./20*(f[j-2] + f[j+2])                  +3./2 *(f[j-1] + f[j+1])                 - 49/18* f[j]        
    return d2f/dx**2

def update_bounds(f):
    for i in range(1,ng+1):
        f[i1-i] = -f[i1+i]
        f[i2+i] = -f[i2-i]
    f[i1]=0
    f[i2]=0
    
    return f

# Initial Condition 

t        = 0 
def doubleExpCutoff(x,vmin,vmax):
    c = (vmax + vmin)/2
    s = (vmax - vmin)/2
    d = (np.exp(-((x - c)/s)**10) - np.exp(-((vmax - c)/s)**10))/         (1 - np.exp(-((vmax - c)/s)**10))
    return d
    
N = 1/x * doubleExpCutoff(x, 1, 50)

plot(x[l1:l2],N[l1:l2])
xscale('log')
yscale('log')
ylim([1e-4,0.2])
xlim([1,50])

alpha_ts   = np.double([0.   , -5./9.  ,-153./128.])
beta_ts    = np.double([1./3., 15./16. ,   8./15. ])

tmax = 1000.
itmax = 100000

courant_advec  = 0.4
courant_diffus = 0.4

dNdt = np.zeros(nx)

ds=0

a = 1/(4*x)
b = 1/4

dt_advection  = courant_advec *  min(dx   / a[l1:l2])
dt_diffusion  = courant_diffus*  dx**2/ b

dt = min([dt_advection,dt_diffusion])
dt_beta_ts = [i * dt for i in beta_ts]

for it in range(itmax):
#
    N = update_bounds(N)
#    
    for itsub in range(3):
        dNdt = alpha_ts[itsub]*dNdt
        ds   = alpha_ts[itsub]*ds
#
        ds=ds+1.
#
        dNdt = dNdt + a[l1:l2]*der(N,x) + b*der2(N,x)
#        
        N[l1:l2] = N[l1:l2] + dt_beta_ts[itsub]*dNdt
        N = update_bounds(N)
#        
        t = t + dt_beta_ts[itsub]*ds
#        
    if (it % 10 == 0): 
        print(it,
          t,
          dt,
          dt_advection,
          dt_diffusion,
          N[l1:l2].max(),N[l1:l2].min())
    
    if ((it == itmax) or t > tmax):
        print('End of simulation at t =',t)
        break
        
plot(x[l1:l2],N[l1:l2])

xlabel(r'$E_B$')
ylabel(r'$N(E_B)$')
B
