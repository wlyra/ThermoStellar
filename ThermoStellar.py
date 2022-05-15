import numpy as np
import pylab as plt
import astropy.constants as const
import sys

#------------------------------------------------
def calc_grid(x0,xn,nx,ng,grid):
    mx = nx + 2*ng    
    i1=ng             # first computational point
    i2=mx-ng-1        # last computational point
    l1=i1
    l2=i2+1

    Lx = xn-x0
    if (grid=='linear'):
        dx = Lx/(nx-1)
        x = np.linspace(x0-3*dx,xn+3*dx,mx)
        dx=np.gradient(x)
        dx_1=1/dx
        dx_tilde=np.zeros(mx)
    elif (grid=='log'):
        #tmp = np.logspace(np.log10(x0),np.log10(xn),nx)
        #k=(np.diff(tmp)/tmp[:-1])[0]
        #x = np.zeros(mx)
        #x[l1:l2]=tmp
        #for i in range(1,4):
        #    x[i1-i] = x[i1]/(1+k)**i
        #    x[i2+i] = x[i2]*(1+k)**i

        xi1 = np.linspace(-ng,nx+ng-1,mx)
        xi1up=nx-1
        xi1lo=0
        a= np.log(xn/x0)/(xi1up-xi1lo)
        b= .5*(xi1up+xi1lo-np.log(xn*x0)/a)
        g1=np.exp(a*(xi1-b))
        g1der1=g1
        g1der2=g1
        g1lo=np.exp(a*(xi1lo-b))
        g1up=np.exp(a*(xi1up-b))
                                       
        x     =x0+Lx*(g1  -  g1lo)/(g1up-g1lo)
        xprim =   Lx*(g1der1*a   )/(g1up-g1lo)
        xprim2=   Lx*(g1der2*a**2)/(g1up-g1lo)

        dx_1=1./xprim
        dx_tilde=-xprim2/xprim**2

    else:
        print("grid function=",grid,"not supported")
        sys.exit()

    return x,dx_1,dx_tilde,mx,l1,l2,i1,i2
#------------------------------------------------
def der(f):
    df = np.zeros(nx)
    for i in range(nx):
        j=i+i1
        df[i] =  -1./60*(f[j-3] - f[j+3]) \
          +3./20*(f[j-2] - f[j+2])        \
          -3./4 *(f[j-1] - f[j+1])        
    return df*dx_1[l1:l2]
#------------------------------------------------
def der2(f):
    d2f = np.zeros(nx)    
    for i in range(nx):
        j=i+i1
        d2f[i] = dx_1[j]**2 * (1./90*(f[j-3] + f[j+3]) \
          -3./20*(f[j-2] + f[j+2])        \
          +3./2 *(f[j-1] + f[j+1])        \
          - 49/18* f[j]        )
    if (grid=='log'):
        df = der(f)      
        d2f = d2f+dx_tilde[l1:l2]*df
    return d2f
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
def ICFilter(x,xInner,xSuppress,xOuter):
  return (np.tanh(x - xInner))*(np.tanh(x - xSuppress) + 1)*np.tanh(-(x - xOuter))/2
#------------------------------------------------
def setproblem(case):

    if (case=='SimpleDiffusion'):
        a = .25/x
        b = np.repeat(.25,mx)
        tmax = 1000                                  # Maximum integration time 
        tout = np.array([1,3,10,30,100,300,1000])   #  Times to plot the data.
        tau_eff=1e30
        lsink = False
    elif (case=='RealDiffusion'):
        lnlambda = 5
        a = np.sqrt(np.pi/2) / (lnlambda*x) * 1.5 * (  x +           1-np.exp(-x) )
        b = np.sqrt(np.pi/2) / (lnlambda*x) * 3.0 * (2*x + x**2 + 2*(1-np.exp(-x)))
        c = np.sqrt(np.pi/2) / (lnlambda*x) * 1.0 * (                1-np.exp(-x) )
        sink_simple = 1 - 3./5*(1-x/xn)**2 - 2./5*(1-2*x/xn)**3
        tau_eff = 1/(c*sink_simple)
        tmax = 100
        tout = np.array([0.3,1,3,10,30,100])
        lsink = True
    else:
        print("Problem case=",case," not implemented")
        sys.exit()
    
    return a,b,tau_eff,lsink,tmax,tout
#------------------------------------------------

# Grid elements.
nx  = 128               # Resolution
grid='log'              # Grid type (log or linear)
 
InitialCondition='Opik' # Opik or DoubleCutoff

case = "RealDiffusion"  # Case to solve (RealDiffusion or SimpleDiffusion)

# The code will calculate until either tmax or itmax is reached.
itmax = 10000000        # Maximum number of timesteps
it_diagnos = 100        # Frequency in timesteps to print to screen

# ----------------------------
# No changes beyond this point
# ----------------------------

ng  = 3                 # number of ghost zones

if (InitialCondition=='DoubleCutoff'):
    x0  = 1    # Grid left limit 
    xn = 50    # Grid left limit
    # call grid
    x,dx_1,dx_tilde,mx,l1,l2,i1,i2 = calc_grid(x0,xn,nx,ng,grid)
    # function 
    N = 1/x * doubleExpCutoff(x, x0, xn)
    # for plotting
    y0=1e-4
    yn=0.2
elif (InitialCondition=='Opik'):
    x0  = 1    # Grid left limit 
    xn = 3000  # Grid left limit
    # call grid
    x,dx_1,dx_tilde,mx,l1,l2,i1,i2 = calc_grid(x0,xn,nx,ng,grid)
    # function 
    xMinIC=10
    N  = 1/x * ICFilter(x, x0, xMinIC, xn)/np.log(xn/xMinIC)
    # for plotting
    y0=1e-4
    yn=0.02
else:
    print("Initial Condition=",InitialCondition," not implemented")
    sys.exit()

N=update_bounds(N)

a,b,tau_eff,lsink,tmax,tout = setproblem(case)

if (lsink==True):
    tau1_eff = 1/tau_eff

# Coefficients for Runge-Kutta (3rd order)

alpha_ts   = np.double([0.   , -5./9.  ,-153./128.])
beta_ts    = np.double([1./3., 15./16. ,   8./15. ])

# Courant numbers 

courant_advec  = 0.4
courant_diffus = 0.4
courant_sink   = np.log(1.3) #30% max change from one timestep to another

dt_advection  = courant_advec *  min((1/dx_1[l1:l2]   )/np.abs(a[l1:l2]))
dt_diffusion  = courant_diffus*  min((1/dx_1[l1:l2]**2)/np.abs(b[l1:l2]))
if (lsink==True):
    dt_sink       = courant_sink  *  min(np.abs(tau_eff[l1:l2]))
    dt = min([dt_advection,dt_diffusion,dt_sink])
    print("--- it --- t --- dt --- dt_advection --- dt_diffusion --- dt_sink --- maxN --- minN")
else:
    dt = min([dt_advection,dt_diffusion])
    print("--- it --- t --- dt --- dt_advection --- dt_diffusion --- maxN --- minN")

dt_beta_ts = [i * dt for i in beta_ts]

dNdt = np.zeros(nx)
ds=0
itout=0
N = update_bounds(N)

# Initialize Time and enter the time loop.

t = 0
plt.plot(x[l1:l2],N[l1:l2],color='black',label=r'$\tau=0$')

for it in range(itmax):
    for itsub in range(3):
        dNdt = alpha_ts[itsub]*dNdt
        ds   = alpha_ts[itsub]*ds
        ds=ds+1.

        if (case=="SimpleDiffusion"):
            
            dNdt = dNdt + a[l1:l2]*der(N) + b[l1:l2]*der2(N)
            
        elif (case=="RealDiffusion"):
            
            dNdt = dNdt - der(a*N) + .5*der2(b*N) - N[l1:l2]*tau1_eff[l1:l2]

        else:
            print("Problem case=",case," not implemented")
            break

        N[l1:l2] = N[l1:l2] + dt_beta_ts[itsub]*dNdt
        
#  Advance time 

        t = t + dt_beta_ts[itsub]*ds

#  Apply boundary conditions for next stage

        N = update_bounds(N)

#  Output to screen 
        
    if (it % it_diagnos == 0):
        if (lsink==True):
            print(it,t,dt,dt_advection,dt_diffusion,dt_sink,N[l1:l2].max(),N[l1:l2].min())
        else:
            print(it,t,dt,dt_advection,dt_diffusion,N[l1:l2].max(),N[l1:l2].min())
            
#  Output to plot

    dt_out = t-tout[itout]
    if ((dt_out > 0) and (dt_out < dt)):
        plt.plot(x[l1:l2],N[l1:l2],label=r'$\tau=$'+str(tout[itout]),linestyle='--')
        itout=itout+1        

    if ((it == itmax) or t > tmax):
        print('End of simulation at t =',t)
        break
        

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$E_B$')
plt.ylabel(r'$N(E_B)$')
plt.ylim([y0,yn])
plt.xlim([x0,xn])
plt.legend()
plt.show()
