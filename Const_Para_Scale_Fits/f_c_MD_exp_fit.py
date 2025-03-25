import numpy as np
import matplotlib.pyplot as plt

# Python file to scale the computational scattering profile (that has error bars) and add a constant paramater 
#to account for the uncertainities during the background subtraction. 
#Also prints the chi value (X) for the comparison between the experimental and computational profiles.

# Loading the experimental data file. First column in the data file should be the q values, 
#second column should be the I(q) values, third column should be the errors in I(q). 
# Change the name within the quotation marks ('') to your experimental file's path.

exp_q, exp_iq, exp_err               = np.loadtxt('EK_16_Back_Sub.txt', unpack = 'True', usecols = (0, 1, 2))

# Loading the computational data file. First column should be the q values (same as experimental q), 
#second column should be the computational I(q), third column should be the errors
# in the computational I(q)
# Change the name within the quotation marks ('') to your computational file's path.

calc_iq, calc_err                    = np.loadtxt('Explicit_Water_q_Iq_Err.txt', unpack = 'True', usecols = (1, 2))
num_points                           = len(exp_q)
max_iter                             = 10000
max_ini_change                       = 100
max_c_ini_per                        = 5
tol_f_rel                            = 0.00005
tol_c_rel                            = 0.00005

for main_iter in range(max_ini_change): 

    mod_ini_change = int(num_points/max_ini_change)   
    f_ini          = exp_iq[main_iter*mod_ini_change]/calc_iq[main_iter*mod_ini_change] 
    c_ini          = -1.0**main_iter*(exp_iq[0]*max_c_ini_per*main_iter)/(100*max_ini_change) 
    para_two_ini   = np.array([[f_ini], [c_ini]])
    search_para    = False

    for i in range(max_iter):
    
        sum_g      = 0.0
        sum_h      = 0.0
        sum_g_derf = 0.0
        sum_g_derc = 0.0
        sum_h_derf = 0.0
        sum_h_derc = 0.0

        for j in range(num_points):

            sum_g      = sum_g + (2*f_ini*calc_iq[j]**2.0 + 2*c_ini*calc_iq[j] - 2*calc_iq[j]*exp_iq[j])/(exp_err[j]**2.0)
            sum_h      = sum_h + (2*c_ini + 2*f_ini*calc_iq[j] - 2*exp_iq[j])/(exp_err[j]**2.0)
            sum_g_derf = sum_g_derf + (2*calc_iq[j]**2.0)/(exp_err[j]**2.0)
            sum_g_derc = sum_g_derc + (2*calc_iq[j])/(exp_err[j]**2.0)
            sum_h_derf = sum_h_derf + (2*calc_iq[j])/(exp_err[j]**2.0)
            sum_h_derc = sum_h_derc + 2/(exp_err[j]**2.0)

        sum_g      = sum_g/num_points
        sum_h      = sum_h/num_points
        sum_g_derf = sum_g_derf/num_points
        sum_g_derc = sum_g_derc/num_points
        sum_h_derf = sum_h_derf/num_points
        sum_h_derc = sum_h_derc/num_points

        jac_mat     = np.array([[sum_g_derf, sum_g_derc], [sum_h_derf, sum_h_derc]])
        det_jac_mat = np.linalg.det(jac_mat)
        if (det_jac_mat == 0):
            break

        func_mat    = np.array([[sum_g], [sum_h]])

        para_two_new = para_two_ini - np.matmul(np.linalg.inv(jac_mat), func_mat)
        f_new = para_two_new[0, 0]
        c_new = para_two_new[1, 0]

        if ((np.abs((f_new - f_ini)*calc_iq[0]/exp_iq[0]) < tol_f_rel) and (np.abs((c_new - c_ini)/exp_iq[0]) < tol_c_rel)):
            search_para = True
            break

        para_two_ini = para_two_new
        f_ini = f_new
        c_ini = c_new 

    if (search_para):
         break

X2 = 0.0

for i in range(num_points):
        X2 = X2 + (((f_new*calc_iq[i] + c_new) - exp_iq[i])/(exp_err[i]))**2.0

X2 = X2/num_points
print('scaling factor f:, c factor:', f_new, c_new)
print('X value for the fit:', np.sqrt(X2))
 
calc_iq_best = calc_iq*f_new + c_new
calc_err_new = calc_err*f_new

fmt = "%20.10f %20.10f %20.10f\n"
out = []

for i in range(num_points):
        a = fmt % (exp_q[i], calc_iq_best[i], calc_err_new[i])
        out.append(a)

# Name of the final scaled computational file. Change the path of the final scaled computational file within the quotation marks ('').

open('Explicit_Water_Scaled_q_Iq.txt', 'w').writelines(out)

# Plotting the computed and experimental scattering profiles.
plt.style.use("paper.mplstyle")
plt.plot(exp_q, calc_iq_best, color= 'green', label = 'Computational SAXS profile', zorder = 3)
plt.fill_between(exp_q, calc_iq_best + calc_err_new, calc_iq_best - calc_err_new, color= 'green',alpha=0.2, zorder = 2)
plt.errorbar(exp_q, exp_iq, yerr = exp_err, color = 'blue', linewidth = 1.0, fmt = 'o', label = 'Experimental SAXS profile', zorder = 1)
#plt.ylim([min_y*0.9, max_y*1.1])
#plt.xlim([0.0, 0.5])
plt.ylabel('I(q)')
plt.xlabel('q')
#plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.legend(loc = 'best')
plt.savefig('Comparison_Computed_Scaled_Profiles.pdf')


