import numpy as np
import matplotlib.pyplot as plt

# Python file to create a scaled computational scattering profile with error bars. Also prints the chi value (X) for the comparison between the experimental and computational profiles.

# Loading the experimental data file. First column in the data file should be the q values, second column should be the I(q) values, third column should be the errors in I(q). 
# Change the name within the quotation marks ('') to your experimental file's path.
q_exp, Iq_exp, err_exp             = np.loadtxt('./EK_16_Back_Sub.txt', unpack = 'True', usecols = (0, 1, 2))
# Loading the computational data file. First column should be the q values (same as experimental q), second column should be the computational I(q), third column should be the errors
# in the computational I(q)
# Change the name within the quotation marks ('') to your computational file's path.
Iq_calc_ff19o, err_calc_ff19o      = np.loadtxt('./Explicit_Water_q_Iq_Err.txt', unpack = 'True', usecols = (1, 2))

# Initialization of relevant variables.
num_points                         = len(q_exp)
sum_1_ff19o = 0.0
sum_2_ff19o = 0.0

for i in range(num_points):
        sum_1_ff19o = sum_1_ff19o + (2*Iq_calc_ff19o[i]*Iq_exp[i])/(err_exp[i]**2.0)
        sum_2_ff19o = sum_2_ff19o + 2*(Iq_calc_ff19o[i]**2.0)/(err_exp[i]**2.0)

scaling_fac_ff19o = sum_1_ff19o/sum_2_ff19o

X2_ff19o = 0

for i in range(num_points):
        X2_ff19o = X2_ff19o + ((scaling_fac_ff19o*Iq_calc_ff19o[i] - Iq_exp[i])/(err_exp[i]))**2.0

X2_ff19o = X2_ff19o/num_points
print('FF19O combination, scaling factor f:', scaling_fac_ff19o, 'X value:', np.sqrt(X2_ff19o))

Iq_calc_new_ff19o = scaling_fac_ff19o * Iq_calc_ff19o
err_calc_new_ff19o = scaling_fac_ff19o * err_calc_ff19o

min_y = np.min(Iq_calc_new_ff19o)
max_y = np.max(Iq_calc_new_ff19o)

fmt = "%20.10f %20.10f %20.10f\n"
out = []

for i in range(num_points):
        a = fmt % (q_exp[i], Iq_calc_new_ff19o[i], err_calc_new_ff19o[i])
        out.append(a)

# Name of the final scaled computational file. Change the path of the final scaled computational file within the quotation marks ('').   
open('Explicit_Water_q_Iq_Scaled.txt', 'w').writelines(out)

# Plotting the computed and experimental scattering profiles.
plt.style.use("paper.mplstyle")
plt.plot(q_exp, Iq_calc_new_ff19o, color= 'green', label = 'Computational SAXS profile', zorder = 3)
plt.fill_between(q_exp, Iq_calc_new_ff19o + err_calc_new_ff19o, Iq_calc_new_ff19o - err_calc_new_ff19o, color= 'green',alpha=0.2, zorder = 2)
plt.errorbar(q_exp, Iq_exp, yerr = err_exp, color = 'blue', linewidth = 1.0, fmt = 'o', label = 'Experimental SAXS profile', zorder = 1)
plt.ylim([min_y*0.9, max_y*1.1])
plt.xlim([0.0, 0.5])
plt.ylabel('I(q)')
plt.xlabel('q')
plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.legend(loc = 'best')
plt.savefig('Comparison_Computed_Scaled_Profiles.pdf')
