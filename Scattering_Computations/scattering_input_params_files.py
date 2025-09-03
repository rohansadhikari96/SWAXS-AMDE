import numpy as np

# Code for users to change the file names and parameters of the SWAXS-AMDE scattering computation.
# To perform the scattering calculation after you have completed the changes here run the command 'python parallel_saxs_all.py'

params = {
    "num_q": 101,                   # Number of q values in the range [0, q_max]
    "q_max": 1.0,                   # Maximum value of q in Angstroms
    "envelope_dim": 7.0,            # Maximum distance from all solute atoms for envelope definition
    "num_orient": 1500,             # Number of orientational vectors for spherical averaging,
    "solv_dens_corr": 1,            # Any value greater than 0 applies solvent density corrections to the scattering calculation
    "corr_solv_dens": 0.334,       # Value for corrected bulk sollvent density in e/A^3
    "guinier_analysis": 0,          # Perform a Guinier analysis calculation to obtain Rg from scattering data
    "fit_exp_data": 1,              # Perform fit of calculated profile to user provided experimental data
    "num_fit_para": 0               # 0 just calculates I(q) at same q values as exp, 1 scales with para f to the experimental data, 2 scales with f and adds a para c for background subtraction uncertanities.
}

# Provide a full path to the file with file names when the code is used on a cluster.
# For example: "traj_prot_solv": '/path/to/file/name_of_protein_water_trajectory_file',

input_files = {
    "traj_prot_solv": './FF19_OPC_WNHMR_10_Frames.nc', # File position and name for protein in solvent trajectory
    "topol_prot_solv": './prot_wat.prmtop', # File position and name for protein in solvent topology
    "traj_bulk_solv": './Waters_OPC_10_Frames.nc', # File position and name for bulk solvent trajectory
    "topol_bulk_solv": './opc_waters.prmtop', #File position and name for bulk solvent topology
    "exp_scatter_data": './EK_16_Back_Sub.txt' # Experimental data to fit against, 3 columns should be q[A-1], I(q), error I(q)
}

# Provide a full path to the file with file names when the code is used on a cluster.

output_files = {
    "comp_iq": 'Explicit_Water_q_Iq.txt', # File position and name for computed I(q) from simulation trajectories
    "guinier_file": 'Guinier_Analysis.txt', # File position and name for computed Rg of simulated system
    "comp_iq_exp_fit": 'Fit_Exp_q_Computed_Iq.txt' # Computed I(q) from simulations fit to experimental data as specified by user
}
