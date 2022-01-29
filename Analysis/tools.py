import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate


def local_max(arr, N = 3):
    '''find local maximums of an array where local is defined as N points on either side'''
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
    '''finds the unique local maximums using local max and where unique is defined as different by the error_tol'''
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


def find_splitting(bifurcation):
    '''takes the result of a bifurcation solve and finds the bifurcation points 2,6,8,16,32,... only limited by the data'''
    indexes = []
    split = True
    split_value = 2
    for index, sub_array in enumerate(bifurcation[:-30]):
        split = True
        
        if len(sub_array) >= split_value and len(indexes) == int(np.log2(split_value)) - 1:
            
            for bi in bifurcation[index: index + 30]:
                if len(bi) < split_value:
                    split = False
            if split:
                 
                indexes.append(index)
                split_value *= 2 

    return np.array(indexes)



class Circuit:
    '''circuit class to streamline the derivative and solving of the ODE, also bifurcation function is nice to have'''
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


def phase_diagram(sol, *args, ax = None, **kwargs):
    '''takes a solution from solve_ivp formatted in the way that the circuit is and plots a phase diagram, quality of life function'''
    if ax is None:
        ax = plt.gca()
    ax.plot(sol.y[0], sol.y[1], *args, **kwargs)

    

    
def generate_Rv_Plots(filename, saveDirectory = None, x_Time = True, x_primeTime=True, phase = True, spectrum = True, printDat=True,plotFigs = True,mainCol = "c",secondCol = "r",markerShape =".",diffPeakThresh = 30,peakN = 500, specN = 500):
    t, negative_x_prime, x = np.genfromtxt(filename, delimiter = ',', unpack = True, skip_header = 12)
    '''takes a filename input formatted according to oscilloscopes output and plots all of:
     - Time serries of x and x'
     - Phase Portrait
     - Spectral Density
     - Prints info about analysis
     With options to turn off each (-), change colour theme, and savefigs. No return'''
    t = t-np.min(t)
    Rv = filename.split('/')[-1][:-4]
    RvNum = Rv[3:]
    returnVal = True
    axs = []

    if x_Time or x_primeTime:
        fig, ax = plt.subplots(figsize = (8,4))
        ax.set(
            title = "Experimental Time Series",#\nRv = " + RvNum + r" [k$\Omega$]",
            xlabel = "Time [s]",
            ylabel = "Voltage [mV]",
            xlim = (0,0.08)
        )
        if x_Time:
            ax.plot(t,x*1000,mainCol+markerShape,markersize = 1,label = "x Voltage")
        if x_primeTime:
            ax.plot(t,-negative_x_prime*1000,secondCol+markerShape,markersize = 1,label = "x' Voltage")
        ax.legend()
        if saveDirectory != None:
            figSaveName = saveDirectory + "/time_series_"+Rv+".png"
            plt.savefig(figSaveName,dpi = 600)
        axs.append(fig)
        if plotFigs: plt.show();
        plt.close()

    if phase:
        fig, ax = plt.subplots(figsize = (6,4))
        ax.set(
            title = "Experimental Phase Portrait",#\nRv = " + RvNum + r" [k$\Omega$]",
            xlabel = "x [mV]",
            ylabel = "x' [mV/s]"
        )
        ax.plot(x*1000,-negative_x_prime*1000,mainCol+markerShape,markersize = 1,alpha = 0.08,label = r"$R_{v}$ = " + RvNum + r" [k$\Omega$]")
        ax.legend()
        if saveDirectory != None:
            figSaveName = saveDirectory + "/phase_portrait_"+Rv+".png"
            plt.savefig(figSaveName,dpi = 600)
        axs.append(fig)
        if plotFigs: plt.show();
        plt.close()

    if spectrum:
        fig, ax = plt.subplots(figsize = (6,4))
        specMaxF = 1600
        ax.set(
            title = "Experimental Spectral Power Density",#\nRv = "+RvNum + r" [k$\Omega$]",
            xlabel = "Frequency [Hz]",
            ylabel = r"Normalized Power Density ($\propto$ $V^{2}$)",
            yscale = "log",
            xlim = (0,specMaxF)
        )
        p = np.abs(np.fft.rfft(x))**2 #xdat is in volts and power = V^2 / R so is perportional up to a resistance
        f = np.linspace(0,1/(np.average(np.diff(t))*2),len(p))
        idx = np.argsort(f)
        powerSpectrumPeaks, spectrumPeakIdx = local_max(p[f<specMaxF], N = specN)
        ax.plot(f[idx], p[idx],linewidth = 1,color = mainCol,label = "Discrete FFT")
        ax.plot(f[spectrumPeakIdx], powerSpectrumPeaks, secondCol+markerShape,markersize = 3,label = "Local Maxs") # spectrumPeakIdx is not of intigers any more
        ax.legend()
        if saveDirectory != None:
            figSaveName = saveDirectory + "/spectral_density_"+Rv+".png"
            plt.savefig(figSaveName,dpi = 600)
        axs.append(fig)
        if plotFigs: plt.show();
        plt.close()

    if printDat:
        dec = 3
        maxs = np.array(unique_maxs(x, N = peakN))
        # Filter out similar maxes determined by threshold above
        peakthresh = (10**(-3))*diffPeakThresh #mV
        tmpIdx = 0
        while tmpIdx < len(maxs):
            tmpmax = maxs[tmpIdx]
            unique = True
            nextIdx = tmpIdx+1
            diffs1 = np.abs(tmpmax-maxs[:tmpIdx])
            diffs2 = np.abs(tmpmax-maxs[nextIdx:])
            if any(np.append(diffs1,diffs2,axis = 0) < peakthresh):
                unique = False
                maxs = np.delete(maxs,tmpIdx)
            tmpIdx += 1
        print("Number of Periods =",len(maxs))
        print("Unique x maxes:",np.around(maxs,dec),"[V]")
        amplitudeX = np.max(x)-np.min(x)
        print("x waveform amplitude: ",np.around(amplitudeX,dec),"[V]")
        amplitudeXprime = np.max(-negative_x_prime)-np.min(-negative_x_prime)
        print("x' waveform amplitude:",np.around(amplitudeXprime,dec),"[V]")
        if spectrum:
            # if data exists print it
            print("Spectral Power Desnitie Peaks:",np.around(f[spectrumPeakIdx],dec),"[Hz]")
        else:
            # if doesnt data exist generate and print it
            specMaxF = 30 # maximum frequency to care about
            p = np.abs(np.fft.rfft(x))**2 #xdat is in volts and power = V^2 / R so is perportional up to a resistance
            f = np.linspace(0,1/(np.average(np.diff(t))*2),len(p))
            idx = np.argsort(f)
            powerSpectrumPeaks, spectrumPeakIdx = local_max(p[f<specMaxF], N = specN)
            print("Spectral Power Desnity Peaks:",np.around(f[spectrumPeakIdx],dec),"[Hz]")
    
    return axs



def phase_diagram_data(filenames, directory = None):
    '''takes filenames as an argument and generate the phase diagrams for them seperately,
    think this is not going to be used but didn't want to delete it just in case'''
    # okay I would just copy paste the code lol if you get it then you should do it, I'm gonna write the save thing, I'll just have it take a directory
    # i can do that if youd like
    axs = []
        
    if type(filenames) != np.array and type(filenames) != list:
        filenames = [filenames]

    for filename in filenames:
        fig, ax = plt.subplots()
        axs.append(ax)
        t, negative_x_prime, x = np.genfromtxt(filename, delimiter = ',', unpack = True, skip_header = 12)
        x_prime = -negative_x_prime
        ax.plot(x, x_prime)
        ax.set(xlabel = 'x', ylabel = "x'", title = filename.split('/')[-1][:-4])
        
        if directory != None:
            plt.savefig(directory + '/phase_' + filename.split('/')[-1][:-4] + '.png')

        plt.show()

        #repeat the above but with fft! or we can make them into subplots not sure how good the phase plot will look tho

    return axs 
