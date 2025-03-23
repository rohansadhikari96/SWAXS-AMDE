# SWAXS-AMDE

SWAXS-AMDE (Small and Wide Angle X-ray Scattering for All Molecular Dynamics Engines) is a Python repository for computing the background subtracted scattering profiles from all-atom MD simulation trajectories. SWAXS-AMDE can handle the binary trajectory files from all of the popular MD engines (NAMD, GROMACS, AMBER, LAMMPS, OpenMM etc.) because it uses the MDTraj Python package to read binary trajectory files. 

![plot](Figure_1.png)

**Figure 1** Graphical representation of the SWAXS-AMDE method. A cavity of the polypeptide and the surrounding solvation layer is carved out from the polypeptide in solvent all-atom trajectory file in (A). A cavity of the same volume is also carved out of the bulk solvent all-atom trajectory file in (B). The scattering amplitudes computed from the polypeptide in solvent simulation box and the bulk solvent simulation box are used to obtain the simulated background subtracted intensities of the system.   

SWAXS-AMDE addresses the challenges frequently found in continuum solvent and other less detailed models of scattering analysis by carefully considering the effect of the density variation of the solvent (due to the presence of the solute) in the computation of the simulated background subtracted intensities. SWAXS-AMDE uses the [MDTraj](https://github.com/mdtraj/mdtraj) library [1] and is hence capable of handling binary trajectory files from all of the main MD engines. MDTraj needs to be installed on the cluster before SWAXS-AMDE can be run by the user. MDTraj can be easily installed using the conda environment on a cluster (webpage for installation of MDTraj - [https://mdtraj.org/1.9.3/installation.html](https://mdtraj.org/1.9.3/installation.html)). The detailed nature of the SWAXS-AMDE program means that a computing cluster is usually required for its use.

SWAXS-AMDE builds on the pioneering theoretical foundations laid by Park et al. [2] who provided the equations to perform an explicit water scattering calculation. In the equations provided by Park et al. the solute had a frozen conformation. Chen and Hub later figured out a way to allow for the thermal fluctuations of the solute while performing an explicit water scattering computation. Their work resulted in the creation of the WAXSIS package [3]. The WAXSIS package is housed on GROMACS and hence requires the trajectory files to be in the xtc format. SWAXS-AMDE performs the explicit water scattering computation while allowing for the thermal fluctuations of the solute. It also has the ability to handle the binary trajectory files in various formats (xtc, dcd, netcdf, dump etc.). 

For now the bulk solvent can only be pure water, but depending on the interest the code will be modified to be able to handle added salt and eventually other kinds of solvents.

Details of the codes provided on this Github page
-------------------------------------------------------------

Scattering_Computations/parallel_saxs_all.py - Main code which performs the scattering computation. Run this code to perform the scattering computation. 'python parallel_saxs_all.py'. The python code requires the MDTraj library.

Scattering_Computations/scattering_input_params_files.py - The file which allows the users to choose the required values for the simulated background subtracted scattering computation. The users can choose the size of the solvation layer. For proteins and water, the recommended hydration layer should be 7 Angstroms away (at least) from all the solute atoms to account for all the density variation in water. The users also have the option of performing a solvent density correction for the scattering computation. Solvent density correction removes the errors from the fluctuations in the bulk density due to finite box sizes and frames in simulations. The users can also specify experimental data and compare the computed scattering calculations to the experimental profile. The users can either choose a scaling only comparison which only finds a factor (f) to uniformly multiply the computed scattering intensities which brings it to the same scale as the experimental profiles. The users can also choose to add a constant parameter (c) to account for the uncertainities in the experimental background subtraction and then compare the computed profiles to the experimental profiles. The users can also choose to not modify the computed profiles and print them as is in [e^2] units but choose to perform the scattering computations at the same q as the experimental data. The users also have the option to perform a Guinier analysis from the computed scattering intensities and obtain the Rg from the computed scattering data. The topology and trajectory files referenced in this code for an example calculation can be downloaded from [zenodo](https://zenodo.org/records/15072930?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk5ZDhlYzFmLWQ0MWYtNDEzMC05NDY1LTBlNDYxYWNhMzYyYyIsImRhdGEiOnt9LCJyYW5kb20iOiJiNDYyMDE4ZjQxZTQ3ZTBiNjE5ZmJkMjk1Y2MwY2ZjOCJ9.XszW2ii4-fQDrXF-wsp9doZfgGnypSlnS_ne6kNXNIG7qXqgrkeg24D5Zp_xt4ymYQjWLVO-HppAbJDfRUsP7g) 

Scaling_Only_Fitting/scale_MD_exp.py - This Python code is relevant if the users need to compute the error bars in the simulated background subtracted intensities. The code scales the computed intensities to the experimental scattering intensities, reports the chi value for the comparison and plots the experimental and the scaled computational I(q). The Scattering_Computations/parallel_saxs_all.py script provides the option to scale the computed scattering intensities to an input experimental profile. This code is hence only relevant if the users want to perform a block averaging analysis first (to obtain the error bars from the I(q) calculated for multiple blocks) and then compare the results to the experimental data. 

Constant_Para_Scale_Fits/f_c_MD_exp_fit.py - This Python code is relevant if the users need to compute the error bars in the simulated background subtracted intensities. The code scales the computed intensities to the experimental scattering intensities and adds a constant parameter to account for the uncertainities in the background subtraction, reports the chi value for the comparison and plots the experimental and the scaled computational I(q). The Scattering_Computations/parallel_saxs_all.py script provides the option to scale the computed scattering intensities and add a constant parameter to account for the uncertainities in the background subtraction and then compare it to the input experimental profile. This code is hence only relevant if the users want to perform a block averaging analysis first (to obtain the error bars from the I(q) calculated for multiple blocks) and then compare the results to the experimental data.




