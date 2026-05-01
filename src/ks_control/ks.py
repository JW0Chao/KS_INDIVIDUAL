import numpy as np

class KS:

    def __init__(self,L=16,N=128,dt=0.05,a_dim=1):
        self.L = L; 
        self.n = N; 
        self.dt = dt; 
        self.x = np.arange(N)*L/N
        self.k = N*np.fft.fftfreq(N)[0:N//2+1]*2*np.pi/L
        self.ik    = 1j*self.k                   # spectral derivative operator
        self.lin   = self.k**2 - self.k**4       # Fourier multipliers for linear term
        self.a_dim = a_dim
        self.B = np.zeros((self.x.size,self.a_dim))
        sig = 0.4
        mid_idx = self.x.size // 2
        gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*((self.x-self.x[mid_idx])/sig)**2)

        # Fixed equispaced actuator center indices: i * N / a_dim.
        # For N=64, a_dim=4 -> [0, 16, 32, 48].
        self.actuator_indices = np.asarray(
            [int(i * self.x.size / self.a_dim) for i in range(self.a_dim)],
            dtype=np.int64,
        )
        for i in range(0, self.a_dim):
            shift = int(self.actuator_indices[i] - mid_idx)
            self.B[:, i] = np.roll(gaus, shift)
            self.B[:, i] = self.B[:, i] / max(self.B[:, i])
    def nlterm(self,u,f):
        # compute tendency from nonlinear term. advection + forcing
        ur = np.fft.irfft(u,axis=-1)
        return -0.5*self.ik*np.fft.rfft(ur**2,axis=-1)+f
    def advance(self,u0,action):
        
        #forcing shape
        self.f0 = np.zeros((self.x.size,1))
        dum = np.zeros((self.x.size,self.a_dim))
        
        for i in range(0,action.size):
            dum[:,i] = self.B[:,i]*action[i]
        self.f0 = np.sum(dum, axis=1)
        
        # semi-implicit third-order runge kutta update.
        # ref: http://journals.ametsoc.org/doi/pdf/10.1175/MWR3214.1
        self.u = np.fft.rfft(u0,axis=-1)
        self.f = np.fft.rfft(self.f0,axis=-1)
        u_save = self.u.copy()
        for n in range(3):
            dt = self.dt/(3-n)
            # explicit RK3 step for nonlinear term
            self.u = u_save + dt*self.nlterm(self.u,self.f)
            # implicit trapezoidal adjustment for linear term
            self.u = (self.u+0.5*self.lin*dt*u_save)/(1.-0.5*self.lin*dt)
        
        
        self.u = np.fft.irfft(self.u,axis=-1)
        return self.u
                
    
