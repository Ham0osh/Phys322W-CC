<Chaotic Circuit>      # top directory for the experiment
   README.txt          # Readme file
   <Analysis>          # Analysis directory
      finalAnalysisScript.py
      myfitfun.py
      preliminaryAnalysisScript.py
   <Data>              # Data directory
      <2018-01-16>
         2018-01-16-descriptive-title.csv
         2018-01-16-new-descriptive-title.csv
      <2018-01-23>
         2018-01-23-descriptive-title.csv
         2018-01-23-new-descriptive-title.csv
<Notes>                # Notes and and other documents
      apparatus-pic.png
      schematic.svg 

# Chaotic Circuit

The purpose of this project is to simulate and observe the behaviour of a chaotic jerk circuit which can be modeled as a third order differential equation with a nonlinear term D(x). This project was made to analyse experimental data from time series voltage measurments and compare it to numerical simulations of the ODE.

Started: Jan 11, 2022
Completed:

### Collaborators:
 - Elliot Wadge
 - Hamish Johnson

### Requirements:
 - Python 3.7
 - Jupyter Lab (Deepnote)
 - Numpy <>
 - SciPy <>
 - MatPlotLib <>
 
 import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate