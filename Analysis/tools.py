import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate


def local_max(arr, N = 3):
    local_maxs = []
    
    #loop through the array
    i = N
    indexes = []
    
    while i < len(arr) - 1 - N:
        
        iterate = 1
        #flag
        local_max = True
        
        for j in range(N):
            if arr[i] < arr[i + j]:
                local_max = False
                iterate = j
                break
                
            elif arr[i] < arr[i - j]:
                local_max = False
                break
            
        if local_max:
            local_maxs.append(arr[i])
            indexes.append(i)
            
        i += iterate
        
    return np.array(local_maxs), np.array(indexes)



def unique_maxs(y: np.array, N = 5, error_tol = 1e-3):
    #get local maximums and sort to make finding repeats simple
    maxs, _ = np.sort(local_max(y, N = N))
    
    try:
        unique_maxs = [maxs[0]]
    except IndexError: 
        print('no maxs')
        return
        
    #remove repeats within a certain error tolerance
    for i in range(1,len(maxs)):
        if np.abs(maxs[i] - unique_maxs[-1]) > error_tol:
            unique_maxs.append(maxs[i])

    return unique_maxs



class Circuit:

    sol = None
    #initializes and stores circuit components
    def __init__(self, Rv, R = 47, R0 = 157, R2_R1 = 6.245, V0 = 0.25, C = 1e-6):
        self.R0 = R0
        self.R = R
        self.ratio_R2_R1 = R2_R1
        self.V0 = V0
        self.Rv = Rv
        self.C = C
        self.Tc = self.R * 1000 * self.C #need to multiply by 1000 because resistance is in kOhm

    #method to return non-linear value of circuit for given v_in
    @staticmethod #static so can be called outside of the class
    def D(v_in, ratio):
        return -ratio*min(v_in,0)


    #method to return the derivatives of the first order system of equations
    #x is x and y is an array of the last position y[0] = x, y[1] = w, y[2] = a
    def derivatives(self,x,y):
        
        x = y[0]
        w = y[1]
        a = y[2]
        
        da = -(self.R/self.Rv)*a - w + self.D(x, self.ratio_R2_R1) - (self.R/self.R0)*self.V0
        dw = a
        dx = w

        return np.array([dx,dw,da])


    #solve method, just a wrapper for solve_ivp with derivative preconfigured that can also plot
    #if you want it to
    def solve(self,tspan, y0, plot = False, phase = False, ax = None, plt_args = [], plt_kwargs = {}, **kwargs):

        sol = solve_ivp(self.derivatives, tspan, y0, **kwargs)
        
        if plot:

            if ax is None:
                ax = plt.gca()


            ax.plot(sol.t, sol.y[0], *plt_args, **plt_kwargs)
            plt.title('x')
            plt.show()
            plt.plot(sol.t, sol.y[1], *plt_args, **plt_kwargs)
            plt.title('w')
            plt.show()
            plt.plot(sol.t, sol.y[2], *plt_args, **plt_kwargs)
            plt.title('a')
            plt.show()

        if phase:
            phase_diagram(sol, *plt_args, **plt_kwargs)

        self.sol = sol
        return sol

    #take an array of Rv values to try and the parameters for solving the IVP
    #as well has how many points qualifies as "local" for a local maximum, and the error tolerance
    #between different maximums
    def bifurcation(self, Rv_arr, tspan, y0, t_eval = None, N = 5, error_tol = 1e-3, **kwargs):
        ret = []
        for Rv in Rv_arr:
            self.Rv = Rv
            sol_x = self.solve(tspan, y0, t_eval = t_eval, **kwargs).y[0]
            unique = unique_maxs(sol_x, error_tol = error_tol)
            ret.append(unique)
        return np.array(ret)


#I think  it makes the most sense to have this outside since it doesn't depend on class at all
#but I added the option to have it called from inside the function when solving with phase = True
def phase_diagram(sol, *args, ax = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(sol.y[0], sol.y[1], *args, **kwargs)

    

    



