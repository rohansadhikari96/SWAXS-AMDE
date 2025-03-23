# SWAXS-AMDE

SWAXS-AMDE (Small and Wide Angle X-ray Scattering for All Molecular Dynamics Engines) is a Python repository for computing the background subtracted scattering profiles from all-atom MD simulation trajectories. SWAXS-AMDE can handle the binary trajectory files from all of the popular MD engines (NAMD, GROMACS, AMBER, LAMMPS, OpenMM etc.) because it uses the MDTraj Python package to read binary trajectory files. 

![plot](Figure_1.png)

**Figure 1** Graphical representation of the SWAXS-AMDE method. A cavity of the polypeptide and the surrounding solvation layer is carved out from the polypeptide in solvent all-atom trajectory file in (A). A cavity of the same volume is also carved out of the bulk solvent all-atom trajectory file in (B). The scattering amplitudes computed from the polypeptide in solvent simulation box and the bulk solvent simulation box are used to obtain the simulated background subtracted intensities of the system.   

SWAXS-AMDE addresses the challenges frequently found in continuum solvent and other less detailed models of scattering analysis by carefully considering the effect of the density variation of the solvent (due to the presence of the solute) in the computation of the simulated background subtracted intensities. SWAXS-AMDE uses the [MDTraj](https://github.com/mdtraj/mdtraj) library and is hence capable of handling binary trajectory files from all of the main MD engines. MDTraj needs to be installed on the cluster before SWAXS-AMDE can be run by the user. MDTraj can be easily installed using the conda environment on a cluster (webpage for installation of MDTraj - [https://mdtraj.org/1.9.3/installation.html](https://mdtraj.org/1.9.3/installation.html)). The detailed nature of the SWAXS-AMDE program means that a computing cluster is usually required for its use.

Details of the codes provided on this Github page
-------------------------------------------------------------
Scaling_Only_Fitting/scale_MD_exp.py - This Python code is relevant if the users need to compute the error bars in the simulated background subtracted intensities. The code scales the computed intensities to the experimental scattering intensities, reports the chi value for the comparison and plots the experimental and the scaled computational I(q). The Scattering_Computations/parallel_saxs_all.py script provides the option to scale the computed scattering intensities to an input experimental profile. This code is hence only relevant if they users want to perform a block averaging analysis from the I(q) calculated for multiple blocks to obtain the error bars for the simulated background subtracted scattering intensities.   


