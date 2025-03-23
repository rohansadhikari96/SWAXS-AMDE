# SWAXS-AMDE

SWAXS-AMDE (Small and Wide Angle X-ray Scattering for All Molecular Dynamics Engines) is a Python repository for computing the background subtracted scattering profiles from all-atom MD simulation trajectories. SWAXS-AMDE can handle the binary trajectory files from all of the popular MD engines (NAMD, GROMACS, AMBER, LAMMPS, OpenMM etc.) because it uses the MDTraj Python package to read binary trajectory files. 

![plot](Figure_1.png)

**Figure 1** Graphical representation of the SWAXS-AMDE method. A cavity of the polymer and the surrounding solvation layer is carved out from the polymer in solvent all-atom trajectory file in (A). A cavity of the same volume is also carved out of the bulk solvent all-atom trajectory file in (B). The scattering amplitudes computed from the polymer in solvent simulation box and the bulk solvent simulation box are used to obtain the simulated background subtracted intensities of the system.   
