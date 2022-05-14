import numpy as np
import pylab as plt
import astropy.constants as const
import sys

#------------------------------------------------
def get_grid_indices(nx,ng):
    mx = nx + 2*ng    
    i1=ng             # first computational point
    i2=mx-ng-1        # last computational point
    l1=i1
    l2=i2+1
    return mx,l1,l2,i1,i2
#------------------------------------------------
def der(f,x):
    n=len(x[l1:l2])
    dx = np.diff(x)[0]
    df = np.zeros(n)
    for i in range(n):
        j=i+i1
        df[i] =  -1./60*(f[j-3] - f[j+3]) \
          +3./20*(f[j-2] - f[j+2])        \
          -3./4 *(f[j-1] - f[j+1])        
    return df/dx
#------------------------------------------------
def der2(f,x):
    n=len(x[l1:l2])
    dx = np.diff(x)[0]
    d2f = np.zeros(n)    
    for i in range(n):
        j=i+i1
        d2f[i] =  1./90*(f[j-3] + f[j+3]) \
          -3./20*(f[j-2] + f[j+2])        \
          +3./2 *(f[j-1] + f[j+1])        \
          - 49/18* f[j]        
    return d2f/dx**2
#------------------------------------------------
def update_bounds(f):
    for i in range(1,ng+1):
        f[i1-i] = -f[i1+i]
        f[i2+i] = -f[i2-i]
    f[i1]=0
    f[i2]=0    
    return f
#------------------------------------------------
def doubleExpCutoff(x,vmin,vmax):
    c = (vmax + vmin)/2
    s = (vmax - vmin)/2
    d = (np.exp(-((x - c)/s)**10) - np.exp(-((vmax - c)/s)**10))/ \
      (1 - np.exp(-((vmax - c)/s)**10))
    return d
#------------------------------------------------

# Grid elements.
nx = 512          # Resolution
x0  = 1           # Grid left limit 
xn  = 50          # Grid right limit

# The code will calculate until either tmax or itmax is reached.

tmax = 1000.      # Maximum integration time 
itmax = 100000    # Maximum number of timesteps
it_diagnos = 100  # Frequency in timesteps to print to screen

#  Times to plot the data.

tout = np.array([1,3,10,30,100,300,1000])

#  Construct grid.

ng = 3
mx,l1,l2,i1,i2 = get_grid_indices(nx,ng)
Lx = xn-x0
dx = Lx/(nx-1)
x = np.linspace(x0-3*dx,xn+3*dx,mx)

# Initial Condition.

N = 1/x * doubleExpCutoff(x, 1, 50)
N = update_bounds(N)

plt.plot(x[l1:l2],N[l1:l2],color='black',label=r'$\tau=0$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$E_B$')
plt.ylabel(r'$N(E_B)$')
plt.ylim([1e-4,0.2])
plt.xlim([x0,xn])

# Coefficients for Runge-Kutta (3rd order)

alpha_ts   = np.double([0.   , -5./9.  ,-153./128.])
beta_ts    = np.double([1./3., 15./16. ,   8./15. ])

# Courant numbers 

courant_advec  = 0.4
courant_diffus = 0.4

a = 1/(4*x)
b = 1/4

dt_advection  = courant_advec *  min(dx   / a[l1:l2])
dt_diffusion  = courant_diffus*  dx**2/ b

dt = min([dt_advection,dt_diffusion])
dt_beta_ts = [i * dt for i in beta_ts]

dNdt = np.zeros(nx)
ds=0
itout=0

print("--- it --- t --- dt --- dt_advection --- dt_diffusion --- maxN --- minN")

# Initialize Time and enter the time loop.

t = 0

for it in range(itmax):
    for itsub in range(3):
        dNdt = alpha_ts[itsub]*dNdt
        ds   = alpha_ts[itsub]*ds
        ds=ds+1.

        dNdt = dNdt + a[l1:l2]*der(N,x) + b*der2(N,x)
        
        N[l1:l2] = N[l1:l2] + dt_beta_ts[itsub]*dNdt
        
#  Advance time 

        t = t + dt_beta_ts[itsub]*ds

#  Apply boundary conditions for next stage

        N = update_bounds(N)

#  Output to screen 
        
    if (it % it_diagnos == 0): 
        print(it,t,dt,dt_advection,dt_diffusion,N[l1:l2].max(),N[l1:l2].min())

#  Output to plot

    dt_out = t-tout[itout]
    if ((dt_out > 0) and (dt_out < dt)):
        plt.plot(x[l1:l2],N[l1:l2],label=r'$\tau=$'+str(tout[itout]),linestyle='--')
        itout=itout+1        

    if ((it == itmax) or t > tmax):
        print('End of simulation at t =',t)
        break
        
plt.legend()
plt.show()
