import os
import numpy as np
import mdtraj as md
import multiprocessing
import random
import scattering_input_params_files
import time

# Run this file to perform the scattering computation after make changes on the 'scattering_input_params_files' file. 

params       = scattering_input_params_files.params
input_files  = scattering_input_params_files.input_files
output_files = scattering_input_params_files.output_files 
    
start_time = time.time()
ncfile  = input_files['traj_prot_solv']
topfile = input_files['topol_prot_solv'] 
traj = md.load(ncfile, top=topfile)
topol = traj.topology
ucelllengths = traj.unitcell_lengths
ucellangles = traj.unitcell_angles
nframes = traj.n_frames
natoms = traj.n_atoms

print('Protein in solvent trajectory loaded')

sulphurs_prot = [atom.index for atom in topol.atoms if (atom.residue.is_protein and atom.name[0] == 'S')]
nSulphurs_prot = len(sulphurs_prot)

oxygens_prot = [atom.index for atom in topol.atoms if (atom.residue.is_protein and atom.name[0] == 'O')]
nOxygens_prot = len(oxygens_prot)

nitrogens_prot = [atom.index for atom in topol.atoms if (atom.residue.is_protein and atom.name[0] == 'N')]
nNitrogens_prot = len(nitrogens_prot)

carbons_prot = [atom.index for atom in topol.atoms if (atom.residue.is_protein and atom.name[0] == 'C')]
nCarbons_prot = len(carbons_prot)

hydrogens_prot = [atom.index for atom in topol.atoms if (atom.residue.is_protein and atom.name[0] == 'H')]
nHydrogens_prot = len(hydrogens_prot)

oxygens_water = [atom.index for atom in topol.atoms if (atom.residue.is_water and atom.name[0] == 'O')]
nOxygens_water = len(oxygens_water)

hydrogens_water = [atom.index for atom in topol.atoms if (atom.residue.is_water and atom.name[0] == 'H')]
nHydrogens_water = len(hydrogens_water)

prot_atoms = [atom.index for atom in topol.atoms if (atom.residue.is_protein)]
num_prot_atoms = len(prot_atoms)

print('Number of protein atoms in the trajectory file is:', num_prot_atoms)
print('Number of waters in the trajectory file is:', nOxygens_water)

prot_bond_pairs_1 = [bond[0].index for bond in topol.bonds if (bond[0].residue.is_protein)]
prot_bond_pairs_2 = [bond[1].index for bond in topol.bonds if (bond[1].residue.is_protein)]
num_prot_bonds = len(prot_bond_pairs_1)

xyz_reimaged = np.zeros([nframes, natoms, 3])

def reimage_frames_prot_solv(frame):
    reimage_frame = np.zeros([natoms, 3])
    reimage_frame[:, 0] = traj.xyz[frame, :, 0] - np.rint(traj.xyz[frame, :, 0]/ucelllengths[frame, 0])*ucelllengths[frame, 0]
    reimage_frame[ :, 1] = traj.xyz[frame, :, 1] - np.rint(traj.xyz[frame, :, 1]/ucelllengths[frame, 1])*ucelllengths[frame, 1]
    reimage_frame[ :, 2] = traj.xyz[frame, :, 2] - np.rint(traj.xyz[frame, :, 2]/ucelllengths[frame, 2])*ucelllengths[frame, 2]

    protein_cross_over_x = False
    protein_cross_over_y = False
    protein_cross_over_z = False

    for bond in range(num_prot_bonds):
        diff_x = np.abs(reimage_frame[prot_bond_pairs_1[bond], 0] - reimage_frame[prot_bond_pairs_2[bond], 0])
        if (np.rint(diff_x/ucelllengths[frame, 0]) > 0):
            protein_cross_over_x = True
            break

    for bond in range(num_prot_bonds):
        diff_y = np.abs(reimage_frame[prot_bond_pairs_1[bond], 1] - reimage_frame[prot_bond_pairs_2[bond], 1])
        if (np.rint(diff_y/ucelllengths[frame, 1]) > 0):
            protein_cross_over_y = True
            break

    for bond in range(num_prot_bonds):
        diff_z = np.abs(reimage_frame[prot_bond_pairs_1[bond], 2] - reimage_frame[prot_bond_pairs_2[bond], 2])
        if (np.rint(diff_z/ucelllengths[frame, 2]) > 0):
            protein_cross_over_z = True
            break

    if (protein_cross_over_x):
        reimage_frame[:, 0] -= ucelllengths[frame, 0]/2
        reimage_frame[:, 0] -= np.rint(reimage_frame[:, 0]/ucelllengths[frame, 0])*ucelllengths[frame, 0]

    if (protein_cross_over_y):
        reimage_frame[:, 1] -= ucelllengths[frame, 1]/2
        reimage_frame[:, 1] -= np.rint(reimage_frame[:, 1]/ucelllengths[frame, 1])*ucelllengths[frame, 1]

    if (protein_cross_over_z):
        reimage_frame[:, 2] -= ucelllengths[frame, 2]/2
        reimage_frame[:, 2] -= np.rint(reimage_frame[:, 2]/ucelllengths[frame, 2])*ucelllengths[frame, 2]

    max_x_protein = np.max(reimage_frame[prot_atoms, 0])
    max_y_protein = np.max(reimage_frame[prot_atoms, 1])
    max_z_protein = np.max(reimage_frame[prot_atoms, 2])

    min_x_protein = np.min(reimage_frame[prot_atoms, 0])
    min_y_protein = np.min(reimage_frame[prot_atoms, 1])
    min_z_protein = np.min(reimage_frame[prot_atoms, 2])

    reimage_frame[:, 0] -= (max_x_protein + min_x_protein)/2.0
    reimage_frame[:, 1] -= (max_y_protein + min_y_protein)/2.0
    reimage_frame[:, 2] -= (max_z_protein + min_z_protein)/2.0

    reimage_frame[:, 0] -= np.rint(reimage_frame[:, 0]/ucelllengths[frame, 0])*ucelllengths[frame, 0]
    reimage_frame[:, 1] -= np.rint(reimage_frame[:, 1]/ucelllengths[frame, 1])*ucelllengths[frame, 1]
    reimage_frame[:, 2] -= np.rint(reimage_frame[:, 2]/ucelllengths[frame, 2])*ucelllengths[frame, 2]

    for oxygen in range(nOxygens_water):

        change_hydro_x1 = np.rint((reimage_frame[oxygens_water[oxygen], 0] - reimage_frame[hydrogens_water[oxygen*2], 0])/ucelllengths[frame, 0])
        reimage_frame[hydrogens_water[oxygen*2], 0] += ucelllengths[frame, 0]*change_hydro_x1
        change_hydro_x2 = np.rint((reimage_frame[oxygens_water[oxygen], 0] - reimage_frame[hydrogens_water[oxygen*2+1], 0])/ucelllengths[frame, 0])
        reimage_frame[hydrogens_water[oxygen*2+1], 0] += ucelllengths[frame, 0]*change_hydro_x2

        change_hydro_y1 = np.rint((reimage_frame[oxygens_water[oxygen], 1] - reimage_frame[hydrogens_water[oxygen*2], 1])/ucelllengths[frame, 1])
        reimage_frame[hydrogens_water[oxygen*2], 1] += ucelllengths[frame, 1]*change_hydro_y1
        change_hydro_y2 = np.rint((reimage_frame[oxygens_water[oxygen], 1] - reimage_frame[hydrogens_water[oxygen*2+1], 1])/ucelllengths[frame, 1])
        reimage_frame[hydrogens_water[oxygen*2+1], 1] += ucelllengths[frame, 1]*change_hydro_y2

        change_hydro_z1 = np.rint((reimage_frame[oxygens_water[oxygen], 2] - reimage_frame[hydrogens_water[oxygen*2], 2])/ucelllengths[frame, 2])
        reimage_frame[hydrogens_water[oxygen*2], 2] += ucelllengths[frame, 2]*change_hydro_z1
        change_hydro_z2 = np.rint((reimage_frame[oxygens_water[oxygen], 2] - reimage_frame[hydrogens_water[oxygen*2+1], 2])/ucelllengths[frame, 2])
        reimage_frame[hydrogens_water[oxygen*2+1], 2] += ucelllengths[frame, 2]*change_hydro_z2

    return reimage_frame

def reimage_frames_bulk_solv(frame):
    reimage_frame = np.zeros([natoms, 3])
    reimage_frame[:, 0] = traj.xyz[frame, :, 0] - np.rint(traj.xyz[frame, :, 0]/ucelllengths[frame, 0])*ucelllengths[frame, 0]
    reimage_frame[ :, 1] = traj.xyz[frame, :, 1] - np.rint(traj.xyz[frame, :, 1]/ucelllengths[frame, 1])*ucelllengths[frame, 1]
    reimage_frame[ :, 2] = traj.xyz[frame, :, 2] - np.rint(traj.xyz[frame, :, 2]/ucelllengths[frame, 2])*ucelllengths[frame, 2]

    for oxygen in range(nOxygens_water):

        change_hydro_x1 = np.rint((reimage_frame[oxygens_water[oxygen], 0] - reimage_frame[hydrogens_water[oxygen*2], 0])/ucelllengths[frame, 0])
        reimage_frame[hydrogens_water[oxygen*2], 0] += ucelllengths[frame, 0]*change_hydro_x1
        change_hydro_x2 = np.rint((reimage_frame[oxygens_water[oxygen], 0] - reimage_frame[hydrogens_water[oxygen*2+1], 0])/ucelllengths[frame, 0])
        reimage_frame[hydrogens_water[oxygen*2+1], 0] += ucelllengths[frame, 0]*change_hydro_x2

        change_hydro_y1 = np.rint((reimage_frame[oxygens_water[oxygen], 1] - reimage_frame[hydrogens_water[oxygen*2], 1])/ucelllengths[frame, 1])
        reimage_frame[hydrogens_water[oxygen*2], 1] += ucelllengths[frame, 1]*change_hydro_y1
        change_hydro_y2 = np.rint((reimage_frame[oxygens_water[oxygen], 1] - reimage_frame[hydrogens_water[oxygen*2+1], 1])/ucelllengths[frame, 1])
        reimage_frame[hydrogens_water[oxygen*2+1], 1] += ucelllengths[frame, 1]*change_hydro_y2

        change_hydro_z1 = np.rint((reimage_frame[oxygens_water[oxygen], 2] - reimage_frame[hydrogens_water[oxygen*2], 2])/ucelllengths[frame, 2])
        reimage_frame[hydrogens_water[oxygen*2], 2] += ucelllengths[frame, 2]*change_hydro_z1
        change_hydro_z2 = np.rint((reimage_frame[oxygens_water[oxygen], 2] - reimage_frame[hydrogens_water[oxygen*2+1], 2])/ucelllengths[frame, 2])
        reimage_frame[hydrogens_water[oxygen*2+1], 2] += ucelllengths[frame, 2]*change_hydro_z2

    return reimage_frame

if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
                reimaged_frames = pool.map(reimage_frames_prot_solv, range(nframes))

num_reimaged_frames = len(reimaged_frames)

for i in range(num_reimaged_frames):
    first_frame, *rest_frame = reimaged_frames
    xyz_reimaged[i, :, :]    = first_frame
    reimaged_frames          = rest_frame

traj = md.Trajectory(xyz = xyz_reimaged, topology = topol, unitcell_lengths = ucelllengths, unitcell_angles = ucellangles)

del xyz_reimaged
del reimaged_frames

num_cos_thet = 72
num_phi  = 144
cos_thet_bin = 2/(num_cos_thet)
phi_bin = (2*np.pi)/(num_phi)
num_points_1 = int(num_cos_thet * num_phi + 2)
num_sp = int(num_phi*num_cos_thet + num_phi*(num_cos_thet - 1) + 2)

sp_x = np.zeros(num_sp)
sp_y = np.zeros(num_sp)
sp_z = np.zeros(num_sp)

sp_x[0] = 0.0
sp_y[0] = 0.0
sp_z[0] = -1.0

count = 1

for a in range(num_cos_thet):
    for b in range(num_phi):
        cos_thet = ((2*a+1)/2)*cos_thet_bin - 1
        sp_x[count] = np.sin(np.arccos(cos_thet))*np.cos(phi_bin*(2*b+1)/2 - np.pi)
        sp_y[count] = np.sin(np.arccos(cos_thet))*np.sin(phi_bin*(2*b+1)/2 - np.pi)
        sp_z[count] = cos_thet
        count = count+1

sp_x[count] = 0.0
sp_y[count] = 0.0
sp_z[count] = 1.0

count = count + 1

for a in range(num_cos_thet - 1):
    for b in range(num_phi):
        sp_x[count] = np.sin(np.arccos(cos_thet_bin*(a+1) - 1))*np.cos(phi_bin*(b+1) - np.pi)
        sp_y[count] = np.sin(np.arccos(cos_thet_bin*(a+1) - 1))*np.sin(phi_bin*(b+1) - np.pi)
        sp_z[count] = (cos_thet_bin*(a+1) - 1)
        count = count+1


if (int(params['fit_exp_data'])):
    q_exp, iq_exp, err_exp = np.loadtxt(input_files['exp_scatter_data'], unpack = 'True', usecols = (0, 1, 2))
    num_exp_data = len(q_exp)

    if (int(params['guinier_analysis'])):

        q_max_guinier = 0.005
        num_guinier = 10
        q_mags_guinier = np.linspace(q_max_guinier, 0.0, num_guinier, endpoint = False)[::-1]
        
        if (q_exp[0] == 0.0):
            q_mag = np.zeros(int(num_exp_data + num_guinier))
            q_mag[:num_exp_data] = q_exp
            q_mag[num_exp_data:] = q_mags_guinier
            q_exp_start = int(0)

        else:
            q_mag = np.zeros(int(num_exp_data + num_guinier + 1))
            q_mag[1:num_exp_data+1] = q_exp
            q_mag[num_exp_data+1:] = q_mags_guinier
            q_exp_start = int(1)

    else:

        if (q_exp[0] == 0.0):
            q_mag = q_exp
            q_exp_start = int(0)

        else:
            q_mag = np.zeros(int(num_exp_data + 1))
            q_mag[1:] = q_exp
            q_exp_start = int(1)

else:

    if (int(params['guinier_analysis'])):
        q_max_guinier = 0.005
        num_guinier = 10
        q_mags_guinier = np.linspace(q_max_guinier, 0.0, num_guinier, endpoint = False)[::-1]
        q_mag_1 = np.linspace(0.0, params['q_max'], int(params['num_q']))
        q_mag = np.zeros(int(params['num_q'] + num_guinier))
        q_mag[:int(params['num_q'])] = q_mag_1
        q_mag[int(params['num_q']):] = q_mags_guinier

    else:
        q_mag = np.linspace(0.0, params['q_max'], int(params['num_q']))

num_q_mag = len(q_mag)

cromer_para_s = np.array([6.90530, 5.20340, 1.43790, 1.58630, 1.46790, 22.2151, 0.253600, 56.1720, 0.866900])
cromer_para_o = np.array([3.04850, 2.28680, 1.54630, 0.867000, 13.2771, 5.70110, 0.323900, 32.9089, 0.250800])
cromer_para_n = np.array([12.2126, 3.13220, 2.01250, 1.16630, 0.005700, 9.89330, 28.9975, 0.582600, -11.529])
cromer_para_c = np.array([2.31000, 1.02000, 1.58860, 0.865000, 20.8439, 10.2075, 0.568700, 51.6512, 0.215600])
cromer_para_h = np.array([0.489918, 0.262003, 0.196767, 0.049879, 20.6593, 7.74039, 49.5519, 2.20159, 0.001305])

cromer_prot_s = np.zeros(num_q_mag)
cromer_prot_o = np.zeros(num_q_mag)
cromer_prot_n = np.zeros(num_q_mag)
cromer_prot_c = np.zeros(num_q_mag)
cromer_prot_h = np.zeros(num_q_mag)
cromer_wat_o = np.zeros(num_q_mag)
cromer_wat_h = np.zeros(num_q_mag)

for i in range(num_q_mag):
    
    for a in range(4):
        cromer_prot_s[i] = cromer_prot_s[i] + cromer_para_s[a]*(np.exp(-1*cromer_para_s[a+4]*(q_mag[i]/(4*np.pi))**2.0))
        cromer_prot_o[i] = cromer_prot_o[i] + cromer_para_o[a]*(np.exp(-1*cromer_para_o[a+4]*(q_mag[i]/(4*np.pi))**2.0))
        cromer_prot_n[i] = cromer_prot_n[i] + cromer_para_n[a]*(np.exp(-1*cromer_para_n[a+4]*(q_mag[i]/(4*np.pi))**2.0))
        cromer_prot_c[i] = cromer_prot_c[i] + cromer_para_c[a]*(np.exp(-1*cromer_para_c[a+4]*(q_mag[i]/(4*np.pi))**2.0))
        cromer_prot_h[i] = cromer_prot_h[i] + cromer_para_h[a]*(np.exp(-1*cromer_para_h[a+4]*(q_mag[i]/(4*np.pi))**2.0))

    cromer_prot_s[i] = cromer_prot_s[i] + cromer_para_s[8]
    cromer_prot_o[i] = cromer_prot_o[i] + cromer_para_o[8]
    cromer_prot_n[i] = cromer_prot_n[i] + cromer_para_n[8]
    cromer_prot_c[i] = cromer_prot_c[i] + cromer_para_c[8]
    cromer_prot_h[i] = cromer_prot_h[i] + cromer_para_h[8]

    cromer_wat_o[i] = cromer_prot_o[i]*(1 + 0.12*np.exp(-1.0*q_mag[i]**2.0/(2*2.2**2.0)))
    cromer_wat_h[i] = cromer_prot_h[i]*(1 - 0.48*np.exp(-1.0*q_mag[i]**2.0/(2*2.2**2.0)))

q_mag = 10.0*q_mag

print('Form factors [f(q)] for all atom types defined')

num_orient = int(params['num_orient'])
q_direc_x = np.zeros(num_orient)
q_direc_y = np.zeros(num_orient)
q_direc_z = np.zeros(num_orient)

for i in range(num_orient):
    loop_var = (2*(i+1) - 1 - num_orient)/num_orient
    loop_thet = np.arccos(loop_var)
    loop_phi = np.sqrt(np.pi*num_orient)*np.arcsin(loop_var)
    q_direc_x[i] = np.sin(loop_thet)*np.cos(loop_phi)
    q_direc_y[i] = np.sin(loop_thet)*np.sin(loop_phi)
    q_direc_z[i] = np.cos(loop_thet)

num_complex_comp = (num_q_mag - 1)*num_orient + 1
q_vec = np.zeros([num_complex_comp, 3], float)

for i in range(0, num_q_mag-1):
    for j in range(num_orient):
        q_vec[1 + i*num_orient + j, 0] = q_mag[i+1]*q_direc_x[j]
        q_vec[1 + i*num_orient + j, 1] = q_mag[i+1]*q_direc_y[j]
        q_vec[1 + i*num_orient + j, 2] = q_mag[i+1]*q_direc_z[j]

cromer_s_comp = np.zeros(num_complex_comp)
cromer_o_comp = np.zeros(num_complex_comp)
cromer_n_comp = np.zeros(num_complex_comp)
cromer_c_comp = np.zeros(num_complex_comp)
cromer_h_comp = np.zeros(num_complex_comp)

cromer_s_comp[0] = cromer_prot_s[0]
cromer_o_comp[0] = cromer_prot_o[0]
cromer_n_comp[0] = cromer_prot_n[0]
cromer_c_comp[0] = cromer_prot_c[0]
cromer_h_comp[0] = cromer_prot_h[0]

for i in range(0, num_q_mag - 1):
    for j in range(num_orient):
        cromer_s_comp[1 + i*num_orient + j] = cromer_prot_s[i+1]
        cromer_o_comp[1 + i*num_orient + j] = cromer_prot_o[i+1]
        cromer_n_comp[1 + i*num_orient + j] = cromer_prot_n[i+1]
        cromer_c_comp[1 + i*num_orient + j] = cromer_prot_c[i+1]
        cromer_h_comp[1 + i*num_orient + j] = cromer_prot_h[i+1]

cromer_o_wat_comp = np.zeros(num_complex_comp)
cromer_h_wat_comp = np.zeros(num_complex_comp)
cromer_o_wat_comp[0] = cromer_wat_o[0]
cromer_h_wat_comp[0] = cromer_wat_h[0]

for i in range(0, num_q_mag - 1):
    for j in range(num_orient):
        cromer_o_wat_comp[1 + i*num_orient + j] = cromer_wat_o[i+1]
        cromer_h_wat_comp[1 + i*num_orient + j] = cromer_wat_h[i+1]

original_rad = params['envelope_dim']/10.0
nframes = traj.n_frames
natoms = traj.n_atoms
xyz_loop = np.zeros([nframes, natoms, 3], float)
dist_arr = np.zeros([nframes, natoms], float)
max_iter = 140
max_iter_2 = 500000
grad_step = 0.00005
tol = 0.00001
rad_vals = np.zeros(num_sp, float)
min_dist_store = np.zeros(num_sp, float)
num_atoms_once = 100

try:
    num_cores = int(os.getenv('SLURM_CPUS_PER_TASK'))
except:
    num_cores = 5

print('Number of cores used is %d' %(num_cores))

num_frames_once = int(num_cores)
floor_frames = np.floor_divide(nframes, num_frames_once)

fmt = "%20.10f %20.10f\n"
fmt_1 = "%20.10f\n"

def def_env_vol(loop_num):
    loop_rad = original_rad
    sp_arr = np.array([sp_x[loop_num]*loop_rad, sp_y[loop_num]* loop_rad, sp_z[loop_num]*loop_rad], float)
    xyz_loop = traj.xyz[:, prot_atoms, :] - sp_arr
    dist_arr = np.sqrt(np.sum(np.multiply(xyz_loop, xyz_loop), axis=2))
    min_dist = np.min(dist_arr)

    for j in range(max_iter):
        if (min_dist < original_rad):
            loop_rad = loop_rad*1.02
        elif (min_dist < (original_rad + tol)):
            rad_vals_func = loop_rad
            min_dist_store_func = min_dist
            break
        else:
            rad_vals_func = loop_rad
            min_dist_store_func = min_dist
            if ((original_rad/min_dist) < 0.98):
                loop_rad = loop_rad*0.98
            else:
                loop_rad = loop_rad*(original_rad/min_dist)

        sp_arr = np.array([sp_x[loop_num]*loop_rad, sp_y[loop_num]* loop_rad, sp_z[loop_num]*loop_rad], float)
        xyz_loop = traj.xyz[:, prot_atoms, :] - sp_arr
        dist_arr = np.sqrt(np.sum(np.multiply(xyz_loop, xyz_loop), axis=2))
        min_dist = np.min(dist_arr)

    if (min_dist_store_func < (original_rad + tol)):
        pass
    else:
        for j in range(1, max_iter_2 + 1):
            min_dist_prev = min_dist
            loop_rad = rad_vals_func - grad_step*j
            sp_arr = np.array([sp_x[loop_num]*loop_rad, sp_y[loop_num]* loop_rad, sp_z[loop_num]*loop_rad], float)
            xyz_loop = traj.xyz[:, prot_atoms, :] - sp_arr
            dist_arr = np.sqrt(np.sum(np.multiply(xyz_loop, xyz_loop), axis=2))
            min_dist = np.min(dist_arr)
            if (min_dist < original_rad):
                rad_vals_func = loop_rad + grad_step
                min_dist_store_func = min_dist_prev
                break

    return rad_vals_func

def solv_dens_corr_prot_solv(frame):
    count_solv_frame = np.zeros([num_sec_solv_dens_corr, bins_per_sec_solv_dens_corr])

    for oxy in range(nOxygens_water):
        oxy_coor     = traj.xyz[frame, oxygens_water[oxy], :]
        rad_o        = np.sqrt(np.sum(np.multiply(oxy_coor, oxy_coor)))

        if (np.absolute(oxy_coor[2]) >= rad_o):
            if (oxy_coor[2] < 0):
                cos_thet_o = -1.0
                phi_o      = (-1 + 2*random.random())*np.pi
            else:
                cos_thet_o = 1.0
                phi_o      = (-1 + 2*random.random())*np.pi
        else:    
            cos_thet_o = oxy_coor[2]/rad_o
            if (np.absolute(oxy_coor[0]) >= np.sqrt(rad_o**2.0 - oxy_coor[2]**2.0)):
                if (oxy_coor[0] < 0):
                    phi_dummy = np.pi
                else:
                    phi_dummy = 0
            else:        
                phi_dummy  = np.arccos(oxy_coor[0]/np.sqrt(rad_o**2.0 - oxy_coor[2]**2.0))

            if (oxy_coor[1] > 0):
                phi_o = phi_dummy
            else:
                phi_o = -1*phi_dummy

        cos_thet_int_o = np.rint((cos_thet_o + 1)/cos_thet_bin)
        phi_int_o  = np.rint((phi_o + np.pi)/phi_bin)

        if (phi_int_o == num_phi):
            phi_int_o = 0

        sec_num_loop = int(cos_thet_int_o*num_phi + phi_int_o)
        rad_dens_corr_loop = rad_dens_corr[sec_num_loop]

        if (rad_o <= rad_dens_corr_loop):
            bin_num = int((rad_o/rad_dens_corr_loop)**3.0*bins_per_sec_solv_dens_corr)
            if (bin_num == bins_per_sec_solv_dens_corr):
                bin_num = bin_num - 1

            count_solv_frame[sec_num_loop, bin_num] += 1
        
    return count_solv_frame

def prot_wat_dens_corr_scattering(sec_num):

    total_amp_real_sec = np.zeros(num_complex_comp, float)
    total_amp_imag_sec = np.zeros(num_complex_comp, float)
    number_elec_loop   = (number_corr_all[sec_num, :]).reshape([bins_per_sec_solv_dens_corr, 1])

    atom_direc_vec = np.transpose(coor_bins_dens_corr[sec_num, :, :])
    atom_dots = np.matmul(q_vec, atom_direc_vec)
    cos_dots = np.cos(atom_dots)
    sin_dots = np.sin(atom_dots)
    total_amp_real_sec = total_amp_real_sec + (np.matmul(cos_dots, number_elec_loop)).reshape(num_complex_comp)
    total_amp_imag_sec = total_amp_imag_sec - (np.matmul(sin_dots, number_elec_loop)).reshape(num_complex_comp)

    return total_amp_real_sec, total_amp_imag_sec

def prot_wat_scattering(frame):

    index_oxygens  = []
    index_hydrogens = []
    count_waters = 0
    total_amp_real_frame = np.zeros(num_complex_comp, float)
    total_amp_imag_frame = np.zeros(num_complex_comp, float)

    floor_prot_sulph = np.floor_divide(nSulphurs_prot, num_atoms_once)
    floor_prot_oxy   = np.floor_divide(nOxygens_prot, num_atoms_once)
    floor_prot_carb  = np.floor_divide(nCarbons_prot, num_atoms_once)
    floor_prot_nitro = np.floor_divide(nNitrogens_prot,num_atoms_once)
    floor_prot_hydro = np.floor_divide(nHydrogens_prot, num_atoms_once)

    if (nSulphurs_prot > 0):

        for i in range(floor_prot_sulph):
            atom_direc_vec = np.transpose(traj.xyz[frame, sulphurs_prot[int(i*num_atoms_once): int((i+1)*num_atoms_once)], :])
            atom_dots      = np.matmul(q_vec, atom_direc_vec)
            cos_dots       = np.cos(atom_dots)
            sin_dots       = np.sin(atom_dots)
            total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_s_comp)
            total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_s_comp)

        atom_direc_vec = np.transpose(traj.xyz[frame, sulphurs_prot[int(floor_prot_sulph*num_atoms_once) : ], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_s_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_s_comp)

    if (nOxygens_prot > 0):

        for i in range(floor_prot_oxy):
            atom_direc_vec = np.transpose(traj.xyz[frame, oxygens_prot[int(i*num_atoms_once): int((i+1)*num_atoms_once)], :])
            atom_dots      = np.matmul(q_vec, atom_direc_vec)
            cos_dots       = np.cos(atom_dots)
            sin_dots       = np.sin(atom_dots)
            total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_o_comp)
            total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_o_comp)

        atom_direc_vec = np.transpose(traj.xyz[frame, oxygens_prot[int(floor_prot_oxy*num_atoms_once) : ], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_o_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_o_comp)

    if (nNitrogens_prot > 0):

        for i in range(floor_prot_nitro):
            atom_direc_vec = np.transpose(traj.xyz[frame, nitrogens_prot[int(i*num_atoms_once): int((i+1)*num_atoms_once)], :])
            atom_dots      = np.matmul(q_vec, atom_direc_vec)
            cos_dots       = np.cos(atom_dots)
            sin_dots       = np.sin(atom_dots)
            total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_n_comp)
            total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_n_comp)

        atom_direc_vec = np.transpose(traj.xyz[frame, nitrogens_prot[int(floor_prot_nitro*num_atoms_once) : ], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_n_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_n_comp)

    if (nCarbons_prot > 0):

        for i in range(floor_prot_carb):
            atom_direc_vec = np.transpose(traj.xyz[frame, carbons_prot[int(i*num_atoms_once): int((i+1)*num_atoms_once)], :])
            atom_dots      = np.matmul(q_vec, atom_direc_vec)
            cos_dots       = np.cos(atom_dots)
            sin_dots       = np.sin(atom_dots)
            total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_c_comp)
            total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_c_comp)

        atom_direc_vec = np.transpose(traj.xyz[frame, carbons_prot[int(floor_prot_carb*num_atoms_once) : ], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_c_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_c_comp)

    if (nHydrogens_prot > 0):

        for i in range(floor_prot_hydro):
            atom_direc_vec = np.transpose(traj.xyz[frame, hydrogens_prot[int(i*num_atoms_once): int((i+1)*num_atoms_once)], :])
            atom_dots      = np.matmul(q_vec, atom_direc_vec)
            cos_dots       = np.cos(atom_dots)
            sin_dots       = np.sin(atom_dots)
            total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_h_comp)
            total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_h_comp)

        atom_direc_vec = np.transpose(traj.xyz[frame, hydrogens_prot[int(floor_prot_hydro*num_atoms_once) : ], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_h_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_h_comp)

    for oxy in range(nOxygens_water):
        water_include = False
        oxy_coor     = traj.xyz[frame, oxygens_water[oxy], :]
        hydro_coor_1 = traj.xyz[frame, hydrogens_water[int(2*oxy)], :]
        hydro_coor_2 = traj.xyz[frame, hydrogens_water[int(2*oxy + 1)], :]
        rad_o        = np.sqrt(np.sum(np.multiply(oxy_coor, oxy_coor)))
        rad_h1       = np.sqrt(np.sum(np.multiply(hydro_coor_1, hydro_coor_1)))
        rad_h2       = np.sqrt(np.sum(np.multiply(hydro_coor_2, hydro_coor_2)))

        if (np.absolute(oxy_coor[2]) >= rad_o):
            if (oxy_coor[2] < 0):
                cos_thet_o = -1.0
                phi_o      = 1.0
            else:
                cos_thet_o = 1.0
                phi_o      = 1.0
        else:    
            cos_thet_o = oxy_coor[2]/rad_o
            if (np.absolute(oxy_coor[0]) >= np.sqrt(rad_o**2.0 - oxy_coor[2]**2.0)):
                if (oxy_coor[0] < 0):
                    phi_dummy = np.pi
                else:
                    phi_dummy = 0
            else:        
                phi_dummy  = np.arccos(oxy_coor[0]/np.sqrt(rad_o**2.0 - oxy_coor[2]**2.0))

            if (oxy_coor[1] > 0):
                phi_o = phi_dummy
            else:
                phi_o = -1*phi_dummy

        cos_thet_int_o = np.rint((cos_thet_o + 1)/cos_thet_bin)
        phi_int_o  = np.rint((phi_o + np.pi)/phi_bin)
        phi_prev_o = phi_int_o - 1

        if (phi_int_o == 0):
            phi_prev_o = num_phi - 1

        if (phi_int_o == num_phi):
            phi_int_o = 0

        if (cos_thet_o == -1):
            rad_array_o = np.array([rads[0]], float)
        elif (cos_thet_o == 1):
            rad_array_o = np.array([rads[num_points_1 - 1]], float)
        elif (cos_thet_int_o == 0):
            rad_val_1     = rads[0]
            rad_val_2     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_prev_o)]  
            rad_val_3     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_int_o)]
            rad_array_o   = np.array([rad_val_1, rad_val_2, rad_val_3])
        elif (cos_thet_int_o == num_cos_thet):
            rad_val_1     = rads[num_points_1 - 1]
            rad_val_2     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_prev_o)]
            rad_val_3     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_int_o)]
            rad_array_o   = np.array([rad_val_1, rad_val_2, rad_val_3])
        else:
            rad_val_1     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_prev_o)]
            rad_val_2     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_int_o)]
            rad_val_3     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_prev_o)]
            rad_val_4     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_int_o)]
            rad_val_5     = rads[int(num_points_1 + (cos_thet_int_o - 1)*num_phi + phi_prev_o)]
            rad_array_o   = np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4, rad_val_5])

        if (np.absolute(hydro_coor_1[2]) >= rad_h1):
            if (hydro_coor_1[2] < 0):
                cos_thet_h1 = -1.0
                phi_h1       = 1.0
            else:
                cos_thet_h1 = 1.0
                phi_h1      = 1.0
        else:    
            cos_thet_h1 = hydro_coor_1[2]/rad_h1
            if (np.absolute(hydro_coor_1[0]) >= np.sqrt(rad_h1**2.0 - hydro_coor_1[2]**2.0)):
                if (hydro_coor_1[0] < 0):
                    phi_dummy = np.pi
                else:
                    phi_dummy = 0
            else:        
                phi_dummy  = np.arccos(hydro_coor_1[0]/np.sqrt(rad_h1**2.0 - hydro_coor_1[2]**2.0))

            if (hydro_coor_1[1] > 0):
                phi_h1 = phi_dummy
            else:
                phi_h1 = -1*phi_dummy

        cos_thet_int_h1 = np.rint((cos_thet_h1 + 1)/cos_thet_bin)
        phi_int_h1      = np.rint((phi_h1 + np.pi)/phi_bin)
        phi_prev_h1     = phi_int_h1 - 1

        if (phi_int_h1 == 0):
            phi_prev_h1 = num_phi - 1

        if (phi_int_h1 == num_phi):
            phi_int_h1 = 0

        if (cos_thet_h1 == -1):
            rad_array_h1 = np.array([rads[0]], float)
        elif (cos_thet_h1 == 1):
            rad_array_h1 = np.array([rads[num_points_1 - 1]], float)
        elif (cos_thet_int_h1 == 0):
            rad_val_1     = rads[0]
            rad_val_2     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_prev_h1)]  
            rad_val_3     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_int_h1)]
            rad_array_h1  = np.array([rad_val_1, rad_val_2, rad_val_3])
        elif (cos_thet_int_h1 == num_cos_thet):
            rad_val_1     = rads[num_points_1 - 1]
            rad_val_2     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_prev_h1)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_int_h1)]
            rad_array_h1  = np.array([rad_val_1, rad_val_2, rad_val_3])
        else:
            rad_val_1     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_prev_h1)]
            rad_val_2     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_int_h1)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_prev_h1)]
            rad_val_4     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_int_h1)]
            rad_val_5     = rads[int(num_points_1 + (cos_thet_int_h1 - 1)*num_phi + phi_prev_h1)]
            rad_array_h1  = np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4, rad_val_5])

        if (np.absolute(hydro_coor_2[2]) >= rad_h2):
            if (hydro_coor_2[2] < 0):
                cos_thet_h2 = -1.0
                phi_h2      = 1.0
            else:
                cos_thet_h2 = 1.0
                phi_h2      = 1.0
        else:    
            cos_thet_h2 = hydro_coor_2[2]/rad_h2
            if (np.absolute(hydro_coor_2[0]) >= np.sqrt(rad_h2**2.0 - hydro_coor_2[2]**2.0)):
                if (hydro_coor_2[0] < 0):
                    phi_dummy = np.pi
                else:
                    phi_dummy = 0
            else:        
                phi_dummy  = np.arccos(hydro_coor_2[0]/np.sqrt(rad_h2**2.0 - hydro_coor_2[2]**2.0))

            if (hydro_coor_2[1] > 0):
                phi_h2 = phi_dummy
            else:
                phi_h2 = -1*phi_dummy

        cos_thet_int_h2 = np.rint((cos_thet_h2 + 1)/cos_thet_bin)
        phi_int_h2      = np.rint((phi_h2 + np.pi)/phi_bin)
        phi_prev_h2     = phi_int_h2 - 1

        if (phi_int_h2 == 0):
            phi_prev_h2 = num_phi - 1

        if (phi_int_h2 == num_phi):
            phi_int_h2 = 0

        if (cos_thet_h2 == -1):
            rad_array_h2 = np.array([rads[0]], float)
        elif (cos_thet_h2 == 1):
            rad_array_h2 = np.array([rads[num_points_1 - 1]], float)
        elif (cos_thet_int_h2 == 0):
            rad_val_1     = rads[0]
            rad_val_2     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_prev_h2)]  
            rad_val_3     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_int_h2)]
            rad_array_h2  = np.array([rad_val_1, rad_val_2, rad_val_3])
        elif (cos_thet_int_h2 == num_cos_thet):
            rad_val_1     = rads[num_points_1 - 1]
            rad_val_2     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_prev_h2)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_int_h2)]
            rad_array_h2  = np.array([rad_val_1, rad_val_2, rad_val_3])
        else:
            rad_val_1     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_prev_h2)]
            rad_val_2     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_int_h2)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_prev_h2)]
            rad_val_4     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_int_h2)]
            rad_val_5     = rads[int(num_points_1 + (cos_thet_int_h2 - 1)*num_phi + phi_prev_h2)]
            rad_array_h2  = np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4, rad_val_5])

        if ((np.min(rad_array_o) >= rad_o) or (np.min(rad_array_h1) >= rad_h1) or (np.min(rad_array_h2) >= rad_h2)):
            water_include = True
        elif ((np.max(rad_array_o) < rad_o) and (np.max(rad_array_h1) < rad_h1) and (np.max(rad_array_h2) < rad_h2)):
            pass
        else:
            if (np.max(rad_array_o) < rad_o):
                pass
            else:
                if (cos_thet_int_o == 0):
                    point_1 = np.array([0.0, 0.0, rad_array_o[0]], float)
                    point_2 = np.array([-phi_bin/2, cos_thet_bin/2, rad_array_o[1]])
                    point_3 = np.array([phi_bin/2,  cos_thet_bin/2, rad_array_o[2]])

                    vector_1 = point_3 - point_1
                    vector_2 = point_2 - point_1

                    cross_vec = np.cross(vector_1, vector_2)
                    if (cross_vec[2] < 0):
                        cross_vec = np.cross(vector_2, vector_1)

                    point_int = np.array([phi_o + np.pi - phi_int_o*phi_bin, cos_thet_o + 1, rad_o])
                    vec_int   = point_int - point_1
                    dot_num  = np.dot(cross_vec, vec_int)

                    if (dot_num < 0.0):
                        water_include = True

                elif (cos_thet_int_o == num_cos_thet):
                    point_1 = np.array([0.0, cos_thet_bin/2, rad_array_o[0]], float)
                    point_2 = np.array([-phi_bin/2, 0.0, rad_array_o[1]])
                    point_3 = np.array([phi_bin/2, 0.0, rad_array_o[2]])

                    vector_1 = point_3 - point_1
                    vector_2 = point_2 - point_1

                    cross_vec = np.cross(vector_1, vector_2)
                    if (cross_vec[2] < 0):
                        cross_vec = np.cross(vector_2, vector_1)

                    point_int = np.array([phi_o + np.pi - phi_int_o*phi_bin, cos_thet_o + 1 - (2*cos_thet_int_o - 1)*cos_thet_bin/2, rad_o])
                    vec_int   = point_int - point_1
                    dot_num  = np.dot(cross_vec, vec_int)

                    if (dot_num < 0.0):
                        water_include = True

                else:
                    point_1 = np.array([0.0, 0.0, rad_array_o[0]], float)
                    point_2 = np.array([phi_bin, 0.0, rad_array_o[1]], float)
                    point_3 = np.array([0.0, cos_thet_bin, rad_array_o[2]], float)
                    point_4 = np.array([phi_bin, cos_thet_bin, rad_array_o[3]], float)
                    point_5 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_o[4]], float)

                    vector_11 = point_2 - point_1
                    vector_21 = point_5 - point_1

                    vector_12 = point_3 - point_1
                    vector_22 = point_5 - point_1

                    vector_13 = point_4 - point_3
                    vector_23 = point_5 - point_3

                    vector_14 = point_4 - point_2 
                    vector_24 = point_5 - point_2 

                    cross_1 = np.cross(vector_11, vector_21)
                    if (cross_1[2] < 0):
                        cross_1 = np.cross(vector_21, vector_11)

                    cross_2 = np.cross(vector_12, vector_22)
                    if (cross_2[2] < 0):
                        cross_2 = np.cross(vector_22, vector_12)

                    cross_3 = np.cross(vector_13, vector_23)
                    if (cross_3[2] < 0):
                        cross_3 = np.cross(vector_23, vector_13)

                    cross_4 = np.cross(vector_14, vector_24)
                    if (cross_4[2] < 0):
                        cross_4 = np.cross(vector_24, vector_14)

                    point_int = np.array([phi_o + np.pi - (2*phi_int_o - 1)*phi_bin/2, cos_thet_o + 1 - (2*cos_thet_int_o - 1)*cos_thet_bin/2, rad_o], float) 

                    vector_int_1 = point_int - point_1
                    vector_int_2 = point_int - point_1
                    vector_int_3 = point_int - point_3
                    vector_int_4 = point_int - point_2

                    dot_1 = np.dot(vector_int_1, cross_1)
                    dot_2 = np.dot(vector_int_2, cross_2)
                    dot_3 = np.dot(vector_int_3, cross_3)
                    dot_4 = np.dot(vector_int_4, cross_4)

                    if ((dot_1 < 0.0) and (dot_2 < 0.0) and (dot_3 < 0.0) and (dot_4 < 0.0)):
                        water_include = True

            if (np.max(rad_array_h1) < rad_h1):
                pass
            else:
                if (water_include):
                    pass
                else:
                    if (cos_thet_int_h1 == 0):
                        point_1 = np.array([0.0, 0.0, rad_array_h1[0]], float)
                        point_2 = np.array([-phi_bin/2, cos_thet_bin/2, rad_array_h1[1]])
                        point_3 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_h1[2]])

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h1 + np.pi - phi_int_h1*phi_bin, cos_thet_h1 + 1, rad_h1])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    elif (cos_thet_int_h1 == num_cos_thet):
                        point_1 = np.array([0.0, cos_thet_bin/2, rad_array_h1[0]], float)
                        point_2 = np.array([-phi_bin/2, 0.0, rad_array_h1[1]], float)
                        point_3 = np.array([phi_bin/2, 0.0, rad_array_h1[2]], float)

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h1 + np.pi - phi_int_h1*phi_bin, cos_thet_h1 + 1 - (2*cos_thet_int_h1 - 1)*cos_thet_bin/2, rad_h1])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    else:
                        point_1 = np.array([0.0, 0.0, rad_array_h1[0]], float)
                        point_2 = np.array([phi_bin, 0.0, rad_array_h1[1]], float)
                        point_3 = np.array([0.0, cos_thet_bin, rad_array_h1[2]], float)
                        point_4 = np.array([phi_bin, cos_thet_bin, rad_array_h1[3]], float)
                        point_5 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_h1[4]], float)

                        vector_11 = point_2 - point_1
                        vector_21 = point_5 - point_1

                        vector_12 = point_3 - point_1
                        vector_22 = point_5 - point_1

                        vector_13 = point_4 - point_3
                        vector_23 = point_5 - point_3

                        vector_14 = point_4 - point_2 
                        vector_24 = point_5 - point_2 

                        cross_1 = np.cross(vector_11, vector_21)
                        if (cross_1[2] < 0):
                            cross_1 = np.cross(vector_21, vector_11)

                        cross_2 = np.cross(vector_12, vector_22)
                        if (cross_2[2] < 0):
                            cross_2 = np.cross(vector_22, vector_12)

                        cross_3 = np.cross(vector_13, vector_23)
                        if (cross_3[2] < 0):
                            cross_3 = np.cross(vector_23, vector_13)

                        cross_4 = np.cross(vector_14, vector_24)
                        if (cross_4[2] < 0):
                            cross_4 = np.cross(vector_24, vector_14)

                        point_int = np.array([phi_h1 + np.pi - (2*phi_int_h1 - 1)*phi_bin/2, cos_thet_h1 + 1 - (2*cos_thet_int_h1 - 1)*cos_thet_bin/2, rad_h1], float) 

                        vector_int_1 = point_int - point_1
                        vector_int_2 = point_int - point_1
                        vector_int_3 = point_int - point_3
                        vector_int_4 = point_int - point_2

                        dot_1 = np.dot(vector_int_1, cross_1)
                        dot_2 = np.dot(vector_int_2, cross_2)
                        dot_3 = np.dot(vector_int_3, cross_3)
                        dot_4 = np.dot(vector_int_4, cross_4)

                        if ((dot_1 < 0.0) and (dot_2 < 0.0) and (dot_3 < 0.0) and (dot_4 < 0.0)):
                            water_include = True
                        
            if (np.max(rad_array_h2) < rad_h2):
                pass
            else:
                if (water_include):
                    pass
                else:
                    if (cos_thet_int_h2 == 0):
                        point_1 = np.array([0.0, 0.0, rad_array_h2[0]], float)
                        point_2 = np.array([-phi_bin/2, cos_thet_bin/2, rad_array_h2[1]])
                        point_3 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_h2[2]])

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h2 + np.pi - phi_int_h2*phi_bin, cos_thet_h2 + 1, rad_h2])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    elif (cos_thet_int_h2 == num_cos_thet):
                        point_1 = np.array([0.0, cos_thet_bin/2, rad_array_h2[0]], float)
                        point_2 = np.array([-phi_bin/2, 0.0, rad_array_h2[1]])
                        point_3 = np.array([phi_bin/2, 0.0, rad_array_h2[2]])

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h2 + np.pi - phi_int_h2*phi_bin, cos_thet_h2 + 1 - (2*cos_thet_int_h2 - 1)*cos_thet_bin/2, rad_h2])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    else:
                        point_1 = np.array([0.0, 0.0, rad_array_h2[0]], float)
                        point_2 = np.array([phi_bin, 0.0, rad_array_h2[1]], float)
                        point_3 = np.array([0.0, cos_thet_bin, rad_array_h2[2]], float)
                        point_4 = np.array([phi_bin, cos_thet_bin, rad_array_h2[3]], float)
                        point_5 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_h2[4]], float)

                        vector_11 = point_2 - point_1
                        vector_21 = point_5 - point_1

                        vector_12 = point_3 - point_1
                        vector_22 = point_5 - point_1

                        vector_13 = point_4 - point_3
                        vector_23 = point_5 - point_3

                        vector_14 = point_4 - point_2 
                        vector_24 = point_5 - point_2 

                        cross_1 = np.cross(vector_11, vector_21)
                        if (cross_1[2] < 0):
                            cross_1 = np.cross(vector_21, vector_11)

                        cross_2 = np.cross(vector_12, vector_22)
                        if (cross_2[2] < 0):
                            cross_2 = np.cross(vector_22, vector_12)

                        cross_3 = np.cross(vector_13, vector_23)
                        if (cross_3[2] < 0):
                            cross_3 = np.cross(vector_23, vector_13)

                        cross_4 = np.cross(vector_14, vector_24)
                        if (cross_4[2] < 0):
                            cross_4 = np.cross(vector_24, vector_14)

                        point_int = np.array([phi_h2 + np.pi - (2*phi_int_h2 - 1)*phi_bin/2, cos_thet_h2 + 1 - (2*cos_thet_int_h2 - 1)*cos_thet_bin/2, rad_h2], float) 

                        vector_int_1 = point_int - point_1
                        vector_int_2 = point_int - point_1
                        vector_int_3 = point_int - point_3
                        vector_int_4 = point_int - point_2

                        dot_1 = np.dot(vector_int_1, cross_1)
                        dot_2 = np.dot(vector_int_2, cross_2)
                        dot_3 = np.dot(vector_int_3, cross_3)
                        dot_4 = np.dot(vector_int_4, cross_4)

                        if ((dot_1 < 0.0) and (dot_2 < 0.0) and (dot_3 < 0.0) and (dot_4 < 0.0)):
                            water_include = True

        if (water_include):
            count_waters = count_waters + 1
            index_oxygens.append(oxygens_water[oxy])
            index_hydrogens.append(hydrogens_water[int(2*oxy)])
            index_hydrogens.append(hydrogens_water[int(2*oxy + 1)])

    num_oxygens   = len(index_oxygens)
    num_hydrogens = len(index_hydrogens)
    floor_oxy     = np.floor_divide(num_oxygens, num_atoms_once)
    floor_hydro   = np.floor_divide(num_hydrogens, num_atoms_once)
    
#    print('Computations for Water Oxygens beginning')
    for i in range(floor_oxy):
        atom_direc_vec = np.transpose(traj.xyz[frame, index_oxygens[int(i*num_atoms_once): int((i+1)*num_atoms_once)], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_o_wat_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_o_wat_comp)

    atom_direc_vec = np.transpose(traj.xyz[frame, index_oxygens[int(floor_oxy*num_atoms_once) : ], :])
    atom_dots      = np.matmul(q_vec, atom_direc_vec)
    cos_dots       = np.cos(atom_dots)
    sin_dots       = np.sin(atom_dots)
    total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_o_wat_comp)
    total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_o_wat_comp)

#    print('Computations for Water Hydrogens beginning')
    for i in range(floor_hydro):
        atom_direc_vec = np.transpose(traj.xyz[frame, index_hydrogens[int(i*num_atoms_once) : int((i+1)*num_atoms_once)], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_h_wat_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_h_wat_comp)

    atom_direc_vec = np.transpose(traj.xyz[frame, index_hydrogens[int(floor_hydro*num_atoms_once) : ], :])
    atom_dots      = np.matmul(q_vec, atom_direc_vec)
    cos_dots       = np.cos(atom_dots)
    sin_dots       = np.sin(atom_dots)
    total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_h_wat_comp)
    total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_h_wat_comp)		

    return total_amp_real_frame, total_amp_imag_frame

def bulk_wat_scattering(frame):
    index_oxygens  = []
    index_hydrogens = []
    count_waters = 0
    total_amp_real_frame = np.zeros(num_complex_comp, float)
    total_amp_imag_frame = np.zeros(num_complex_comp, float)

    for oxy in range(nOxygens_water):
        water_include = False
        oxy_coor     = traj.xyz[frame, oxygens_water[oxy], :]
        hydro_coor_1 = traj.xyz[frame, hydrogens_water[int(2*oxy)], :]
        hydro_coor_2 = traj.xyz[frame, hydrogens_water[int(2*oxy + 1)], :]
        rad_o        = np.sqrt(np.sum(np.multiply(oxy_coor, oxy_coor)))
        rad_h1       = np.sqrt(np.sum(np.multiply(hydro_coor_1, hydro_coor_1)))
        rad_h2       = np.sqrt(np.sum(np.multiply(hydro_coor_2, hydro_coor_2)))

        if (np.absolute(oxy_coor[2]) >= rad_o):
            if (oxy_coor[2] < 0):
                cos_thet_o = -1.0
                phi_o      = 1.0
            else:
                cos_thet_o = 1.0
                phi_o      = 1.0
        else:    
            cos_thet_o = oxy_coor[2]/rad_o
            if (np.absolute(oxy_coor[0]) >= np.sqrt(rad_o**2.0 - oxy_coor[2]**2.0)):
                if (oxy_coor[0] < 0):
                    phi_dummy = np.pi
                else:
                    phi_dummy = 0
            else:        
                phi_dummy  = np.arccos(oxy_coor[0]/np.sqrt(rad_o**2.0 - oxy_coor[2]**2.0))

            if (oxy_coor[1] > 0):
                phi_o = phi_dummy
            else:
                phi_o = -1*phi_dummy

        cos_thet_int_o = np.rint((cos_thet_o + 1)/cos_thet_bin)
        phi_int_o  = np.rint((phi_o + np.pi)/phi_bin)
        phi_prev_o = phi_int_o - 1

        if (phi_int_o == 0):
            phi_prev_o = num_phi - 1

        if (phi_int_o == num_phi):
            phi_int_o = 0

        if (cos_thet_o == -1):
            rad_array_o = np.array([rads[0]], float)
        elif (cos_thet_o == 1):
            rad_array_o = np.array([rads[num_points_1 - 1]], float)
        elif (cos_thet_int_o == 0):
            rad_val_1     = rads[0]
            rad_val_2     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_prev_o)]  
            rad_val_3     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_int_o)]
            rad_array_o   = np.array([rad_val_1, rad_val_2, rad_val_3])
        elif (cos_thet_int_o == num_cos_thet):
            rad_val_1     = rads[num_points_1 - 1]
            rad_val_2     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_prev_o)]
            rad_val_3     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_int_o)]
            rad_array_o   = np.array([rad_val_1, rad_val_2, rad_val_3])
        else:
            rad_val_1     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_prev_o)]
            rad_val_2     = rads[int(1 + (cos_thet_int_o - 1)*num_phi + phi_int_o)]
            rad_val_3     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_prev_o)]
            rad_val_4     = rads[int(1 + (cos_thet_int_o)*num_phi + phi_int_o)]
            rad_val_5     = rads[int(num_points_1 + (cos_thet_int_o - 1)*num_phi + phi_prev_o)]
            rad_array_o   = np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4, rad_val_5])

        if (np.absolute(hydro_coor_1[2]) >= rad_h1):
            if (hydro_coor_1[2] < 0):
                cos_thet_h1 = -1.0
                phi_h1       = 1.0
            else:
                cos_thet_h1 = 1.0
                phi_h1      = 1.0
        else:    
            cos_thet_h1 = hydro_coor_1[2]/rad_h1
            if (np.absolute(hydro_coor_1[0]) >= np.sqrt(rad_h1**2.0 - hydro_coor_1[2]**2.0)):
                if (hydro_coor_1[0] < 0):
                    phi_dummy = np.pi
                else:
                    phi_dummy = 0
            else:        
                phi_dummy  = np.arccos(hydro_coor_1[0]/np.sqrt(rad_h1**2.0 - hydro_coor_1[2]**2.0))

            if (hydro_coor_1[1] > 0):
                phi_h1 = phi_dummy
            else:
                phi_h1 = -1*phi_dummy

        cos_thet_int_h1 = np.rint((cos_thet_h1 + 1)/cos_thet_bin)
        phi_int_h1      = np.rint((phi_h1 + np.pi)/phi_bin)
        phi_prev_h1     = phi_int_h1 - 1

        if (phi_int_h1 == 0):
            phi_prev_h1 = num_phi - 1

        if (phi_int_h1 == num_phi):
            phi_int_h1 = 0

        if (cos_thet_h1 == -1):
            rad_array_h1 = np.array([rads[0]], float)
        elif (cos_thet_h1 == 1):
            rad_array_h1 = np.array([rads[num_points_1 - 1]], float)
        elif (cos_thet_int_h1 == 0):
            rad_val_1     = rads[0]
            rad_val_2     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_prev_h1)]  
            rad_val_3     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_int_h1)]
            rad_array_h1  = np.array([rad_val_1, rad_val_2, rad_val_3])
        elif (cos_thet_int_h1 == num_cos_thet):
            rad_val_1     = rads[num_points_1 - 1]
            rad_val_2     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_prev_h1)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_int_h1)]
            rad_array_h1  = np.array([rad_val_1, rad_val_2, rad_val_3])
        else:
            rad_val_1     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_prev_h1)]
            rad_val_2     = rads[int(1 + (cos_thet_int_h1 - 1)*num_phi + phi_int_h1)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_prev_h1)]
            rad_val_4     = rads[int(1 + (cos_thet_int_h1)*num_phi + phi_int_h1)]
            rad_val_5     = rads[int(num_points_1 + (cos_thet_int_h1 - 1)*num_phi + phi_prev_h1)]
            rad_array_h1  = np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4, rad_val_5])

        if (np.absolute(hydro_coor_2[2]) >= rad_h2):
            if (hydro_coor_2[2] < 0):
                cos_thet_h2 = -1.0
                phi_h2      = 1.0
            else:
                cos_thet_h2 = 1.0
                phi_h2      = 1.0
        else:    
            cos_thet_h2 = hydro_coor_2[2]/rad_h2
            if (np.absolute(hydro_coor_2[0]) >= np.sqrt(rad_h2**2.0 - hydro_coor_2[2]**2.0)):
                if (hydro_coor_2[0] < 0):
                    phi_dummy = np.pi
                else:
                    phi_dummy = 0
            else:        
                phi_dummy  = np.arccos(hydro_coor_2[0]/np.sqrt(rad_h2**2.0 - hydro_coor_2[2]**2.0))

            if (hydro_coor_2[1] > 0):
                phi_h2 = phi_dummy
            else:
                phi_h2 = -1*phi_dummy

        cos_thet_int_h2 = np.rint((cos_thet_h2 + 1)/cos_thet_bin)
        phi_int_h2      = np.rint((phi_h2 + np.pi)/phi_bin)
        phi_prev_h2 = phi_int_h2 - 1

        if (phi_int_h2 == 0):
            phi_prev_h2 = num_phi - 1

        if (phi_int_h2 == num_phi):
            phi_int_h2 = 0

        if (cos_thet_h2 == -1):
            rad_array_h2 = np.array([rads[0]], float)
        elif (cos_thet_h2 == 1):
            rad_array_h2 = np.array([rads[num_points_1 - 1]], float)
        elif (cos_thet_int_h2 == 0):
            rad_val_1     = rads[0]
            rad_val_2     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_prev_h2)]  
            rad_val_3     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_int_h2)]
            rad_array_h2  = np.array([rad_val_1, rad_val_2, rad_val_3])
        elif (cos_thet_int_h2 == num_cos_thet):
            rad_val_1     = rads[num_points_1 - 1]
            rad_val_2     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_prev_h2)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_int_h2)]
            rad_array_h2  = np.array([rad_val_1, rad_val_2, rad_val_3])
        else:
            rad_val_1     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_prev_h2)]
            rad_val_2     = rads[int(1 + (cos_thet_int_h2 - 1)*num_phi + phi_int_h2)]
            rad_val_3     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_prev_h2)]
            rad_val_4     = rads[int(1 + (cos_thet_int_h2)*num_phi + phi_int_h2)]
            rad_val_5     = rads[int(num_points_1 + (cos_thet_int_h2 - 1)*num_phi + phi_prev_h2)]
            rad_array_h2  = np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4, rad_val_5])

        if ((np.min(rad_array_o) >= rad_o) or (np.min(rad_array_h1) >= rad_h1) or (np.min(rad_array_h2) >= rad_h2)):
            water_include = True
        elif ((np.max(rad_array_o) < rad_o) and (np.max(rad_array_h1) < rad_h1) and (np.max(rad_array_h2) < rad_h2)):
            pass
        else:
            if (np.max(rad_array_o) < rad_o):
                pass
            else:
                if (cos_thet_int_o == 0):
                    point_1 = np.array([0.0, 0.0, rad_array_o[0]], float)
                    point_2 = np.array([-phi_bin/2, cos_thet_bin/2, rad_array_o[1]])
                    point_3 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_o[2]])

                    vector_1 = point_3 - point_1
                    vector_2 = point_2 - point_1

                    cross_vec = np.cross(vector_1, vector_2)
                    if (cross_vec[2] < 0):
                        cross_vec = np.cross(vector_2, vector_1)

                    point_int = np.array([phi_o + np.pi - phi_int_o*phi_bin, cos_thet_o + 1, rad_o])
                    vec_int   = point_int - point_1
                    dot_num  = np.dot(cross_vec, vec_int)

                    if (dot_num < 0.0):
                        water_include = True

                elif (cos_thet_int_o == num_cos_thet):
                    point_1 = np.array([0.0, cos_thet_bin/2, rad_array_o[0]], float)
                    point_2 = np.array([-phi_bin/2, 0.0, rad_array_o[1]])
                    point_3 = np.array([phi_bin/2, 0.0, rad_array_o[2]])

                    vector_1 = point_3 - point_1
                    vector_2 = point_2 - point_1

                    cross_vec = np.cross(vector_1, vector_2)
                    if (cross_vec[2] < 0):
                        cross_vec = np.cross(vector_2, vector_1)

                    point_int = np.array([phi_o + np.pi - phi_int_o*phi_bin, cos_thet_o + 1 - (2*cos_thet_int_o - 1)*cos_thet_bin/2, rad_o])
                    vec_int   = point_int - point_1
                    dot_num  = np.dot(cross_vec, vec_int)

                    if (dot_num < 0.0):
                        water_include = True

                else:
                    point_1 = np.array([0.0, 0.0, rad_array_o[0]], float)
                    point_2 = np.array([phi_bin, 0.0, rad_array_o[1]], float)
                    point_3 = np.array([0.0, cos_thet_bin, rad_array_o[2]], float)
                    point_4 = np.array([phi_bin, cos_thet_bin, rad_array_o[3]], float)
                    point_5 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_o[4]], float)

                    vector_11 = point_2 - point_1
                    vector_21 = point_5 - point_1

                    vector_12 = point_3 - point_1
                    vector_22 = point_5 - point_1

                    vector_13 = point_4 - point_3
                    vector_23 = point_5 - point_3

                    vector_14 = point_4 - point_2 
                    vector_24 = point_5 - point_2 

                    cross_1 = np.cross(vector_11, vector_21)
                    if (cross_1[2] < 0):
                        cross_1 = np.cross(vector_21, vector_11)

                    cross_2 = np.cross(vector_12, vector_22)
                    if (cross_2[2] < 0):
                        cross_2 = np.cross(vector_22, vector_12)

                    cross_3 = np.cross(vector_13, vector_23)
                    if (cross_3[2] < 0):
                        cross_3 = np.cross(vector_23, vector_13)

                    cross_4 = np.cross(vector_14, vector_24)
                    if (cross_4[2] < 0):
                        cross_4 = np.cross(vector_24, vector_14)

                    point_int = np.array([phi_o + np.pi - (2*phi_int_o - 1)*phi_bin/2, cos_thet_o + 1 - (2*cos_thet_int_o - 1)*cos_thet_bin/2, rad_o], float) 

                    vector_int_1 = point_int - point_1
                    vector_int_2 = point_int - point_1
                    vector_int_3 = point_int - point_3
                    vector_int_4 = point_int - point_2

                    dot_1 = np.dot(vector_int_1, cross_1)
                    dot_2 = np.dot(vector_int_2, cross_2)
                    dot_3 = np.dot(vector_int_3, cross_3)
                    dot_4 = np.dot(vector_int_4, cross_4)

                    if ((dot_1 < 0.0) and (dot_2 < 0.0) and (dot_3 < 0.0) and (dot_4 < 0.0)):
                        water_include = True

            if (np.max(rad_array_h1) < rad_h1):
                pass
            else:
                if (water_include):
                    pass
                else:
                    if (cos_thet_int_h1 == 0):
                        point_1 = np.array([0.0, 0.0, rad_array_h1[0]], float)
                        point_2 = np.array([-phi_bin/2, cos_thet_bin/2, rad_array_h1[1]])
                        point_3 = np.array([phi_bin/2,  cos_thet_bin/2, rad_array_h1[2]])

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h1 + np.pi - phi_int_h1*phi_bin, cos_thet_h1 + 1, rad_h1])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    elif (cos_thet_int_h1 == num_cos_thet):
                        point_1 = np.array([0.0, cos_thet_bin/2, rad_array_h1[0]], float)
                        point_2 = np.array([-phi_bin/2, 0.0, rad_array_h1[1]])
                        point_3 = np.array([phi_bin/2, 0.0, rad_array_h1[2]])

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h1 + np.pi - phi_int_h1*phi_bin, cos_thet_h1 + 1 - (2*cos_thet_int_h1 - 1)*cos_thet_bin/2, rad_h1])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    else:
                        point_1 = np.array([0.0, 0.0, rad_array_h1[0]], float)
                        point_2 = np.array([phi_bin, 0.0, rad_array_h1[1]], float)
                        point_3 = np.array([0.0, cos_thet_bin, rad_array_h1[2]], float)
                        point_4 = np.array([phi_bin, cos_thet_bin, rad_array_h1[3]], float)
                        point_5 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_h1[4]], float)

                        vector_11 = point_2 - point_1
                        vector_21 = point_5 - point_1

                        vector_12 = point_3 - point_1
                        vector_22 = point_5 - point_1

                        vector_13 = point_4 - point_3
                        vector_23 = point_5 - point_3

                        vector_14 = point_4 - point_2 
                        vector_24 = point_5 - point_2 

                        cross_1 = np.cross(vector_11, vector_21)
                        if (cross_1[2] < 0):
                            cross_1 = np.cross(vector_21, vector_11)

                        cross_2 = np.cross(vector_12, vector_22)
                        if (cross_2[2] < 0):
                            cross_2 = np.cross(vector_22, vector_12)

                        cross_3 = np.cross(vector_13, vector_23)
                        if (cross_3[2] < 0):
                            cross_3 = np.cross(vector_23, vector_13)

                        cross_4 = np.cross(vector_14, vector_24)
                        if (cross_4[2] < 0):
                            cross_4 = np.cross(vector_24, vector_14)

                        point_int = np.array([phi_h1 + np.pi - (2*phi_int_h1 - 1)*phi_bin/2, cos_thet_h1 + 1 - (2*cos_thet_int_h1 - 1)*cos_thet_bin/2, rad_h1], float) 

                        vector_int_1 = point_int - point_1
                        vector_int_2 = point_int - point_1
                        vector_int_3 = point_int - point_3
                        vector_int_4 = point_int - point_2

                        dot_1 = np.dot(vector_int_1, cross_1)
                        dot_2 = np.dot(vector_int_2, cross_2)
                        dot_3 = np.dot(vector_int_3, cross_3)
                        dot_4 = np.dot(vector_int_4, cross_4)

                        if ((dot_1 < 0.0) and (dot_2 < 0.0) and (dot_3 < 0.0) and (dot_4 < 0.0)):
                            water_include = True
                        
            if (np.max(rad_array_h2) < rad_h2):
                pass
            else:
                if (water_include):
                    pass
                else:
                    if (cos_thet_int_h2 == 0):
                        point_1 = np.array([0.0, 0.0, rad_array_h2[0]], float)
                        point_2 = np.array([-phi_bin/2, cos_thet_bin/2, rad_array_h2[1]])
                        point_3 = np.array([phi_bin/2,  cos_thet_bin/2, rad_array_h2[2]])

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h2 + np.pi - phi_int_h2*phi_bin, cos_thet_h2 + 1, rad_h2])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    elif (cos_thet_int_h2 == num_cos_thet):
                        point_1 = np.array([0.0, cos_thet_bin/2, rad_array_h2[0]], float)
                        point_2 = np.array([-phi_bin/2, 0.0, rad_array_h2[1]])
                        point_3 = np.array([phi_bin/2, 0.0, rad_array_h2[2]])

                        vector_1 = point_3 - point_1
                        vector_2 = point_2 - point_1

                        cross_vec = np.cross(vector_1, vector_2)
                        if (cross_vec[2] < 0):
                            cross_vec = np.cross(vector_2, vector_1)

                        point_int = np.array([phi_h2 + np.pi - phi_int_h2*phi_bin, cos_thet_h2 + 1 - (2*cos_thet_int_h2 - 1)*cos_thet_bin/2, rad_h2])
                        vec_int   = point_int - point_1
                        dot_num  = np.dot(cross_vec, vec_int)

                        if (dot_num < 0.0):
                            water_include = True

                    else:
                        point_1 = np.array([0.0, 0.0, rad_array_h2[0]], float)
                        point_2 = np.array([phi_bin, 0.0, rad_array_h2[1]], float)
                        point_3 = np.array([0.0, cos_thet_bin, rad_array_h2[2]], float)
                        point_4 = np.array([phi_bin, cos_thet_bin, rad_array_h2[3]], float)
                        point_5 = np.array([phi_bin/2, cos_thet_bin/2, rad_array_h2[4]], float)

                        vector_11 = point_2 - point_1
                        vector_21 = point_5 - point_1

                        vector_12 = point_3 - point_1
                        vector_22 = point_5 - point_1

                        vector_13 = point_4 - point_3
                        vector_23 = point_5 - point_3

                        vector_14 = point_4 - point_2 
                        vector_24 = point_5 - point_2 

                        cross_1 = np.cross(vector_11, vector_21)
                        if (cross_1[2] < 0):
                            cross_1 = np.cross(vector_21, vector_11)

                        cross_2 = np.cross(vector_12, vector_22)
                        if (cross_2[2] < 0):
                            cross_2 = np.cross(vector_22, vector_12)

                        cross_3 = np.cross(vector_13, vector_23)
                        if (cross_3[2] < 0):
                            cross_3 = np.cross(vector_23, vector_13)

                        cross_4 = np.cross(vector_14, vector_24)
                        if (cross_4[2] < 0):
                            cross_4 = np.cross(vector_24, vector_14)

                        point_int = np.array([phi_h2 + np.pi - (2*phi_int_h2 - 1)*phi_bin/2, cos_thet_h2 + 1 - (2*cos_thet_int_h2 - 1)*cos_thet_bin/2, rad_h2], float) 

                        vector_int_1 = point_int - point_1
                        vector_int_2 = point_int - point_1
                        vector_int_3 = point_int - point_3
                        vector_int_4 = point_int - point_2

                        dot_1 = np.dot(vector_int_1, cross_1)
                        dot_2 = np.dot(vector_int_2, cross_2)
                        dot_3 = np.dot(vector_int_3, cross_3)
                        dot_4 = np.dot(vector_int_4, cross_4)

                        if ((dot_1 < 0.0) and (dot_2 < 0.0) and (dot_3 < 0.0) and (dot_4 < 0.0)):
                            water_include = True

        if (water_include):
            count_waters = count_waters + 1
            index_oxygens.append(oxygens_water[oxy])
            index_hydrogens.append(hydrogens_water[int(2*oxy)])
            index_hydrogens.append(hydrogens_water[int(2*oxy + 1)])


    num_oxygens   = len(index_oxygens)
    num_hydrogens = len(index_hydrogens)
    floor_oxy     = np.floor_divide(num_oxygens, num_atoms_once)
    floor_hydro   = np.floor_divide(num_hydrogens, num_atoms_once)
    
    for i in range(floor_oxy):
        atom_direc_vec = np.transpose(traj.xyz[frame, index_oxygens[int(i*num_atoms_once): int((i+1)*num_atoms_once)], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_o_wat_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_o_wat_comp)

    atom_direc_vec = np.transpose(traj.xyz[frame, index_oxygens[int(floor_oxy*num_atoms_once) : ], :])
    atom_dots      = np.matmul(q_vec, atom_direc_vec)
    cos_dots       = np.cos(atom_dots)
    sin_dots       = np.sin(atom_dots)
    total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_o_wat_comp)
    total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_o_wat_comp)

    for i in range(floor_hydro):
        atom_direc_vec = np.transpose(traj.xyz[frame, index_hydrogens[int(i*num_atoms_once) : int((i+1)*num_atoms_once)], :])
        atom_dots      = np.matmul(q_vec, atom_direc_vec)
        cos_dots       = np.cos(atom_dots)
        sin_dots       = np.sin(atom_dots)
        total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_h_wat_comp)
        total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_h_wat_comp)

    atom_direc_vec = np.transpose(traj.xyz[frame, index_hydrogens[int(floor_hydro*num_atoms_once) : ], :])
    atom_dots      = np.matmul(q_vec, atom_direc_vec)
    cos_dots       = np.cos(atom_dots)
    sin_dots       = np.sin(atom_dots)
    total_amp_real_frame = total_amp_real_frame + np.multiply(np.transpose(np.sum(cos_dots, axis = 1)), cromer_h_wat_comp)
    total_amp_imag_frame = total_amp_imag_frame - np.multiply(np.transpose(np.sum(sin_dots, axis = 1)), cromer_h_wat_comp)	

    return total_amp_real_frame, total_amp_imag_frame

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        rads = pool.map(def_env_vol, range(num_sp))

print('%f Angstrom envelope defined around collective solute atoms from all frames' %(params['envelope_dim']))

num_entries_env = len(rads)
min_boxlength_x = np.min(ucelllengths[:, 0])
min_boxlength_y = np.min(ucelllengths[:, 1])
min_boxlength_z = np.min(ucelllengths[:, 2])
env_cross = False

for i in range(num_entries_env):
    if (np.absolute(sp_x[i]*rads[i]) >= min_boxlength_x/2):
        env_cross = True
        break

    if (np.absolute(sp_y[i]*rads[i]) >= min_boxlength_y/2):
        env_cross = True
        break   

    if (np.absolute(sp_z[i]*rads[i]) >= min_boxlength_z/2):
        env_cross = True
        break

if (env_cross):
    print('Warning! Dimensions of envelope excede the boundaries of the protein in solvent simulation box. Consider increasing the size of the simulation box')
else:
    print('Envelope dimensions within the boundaries of the protein in solvent simulation box')
    
total_amp_prot_wat_dens_corr_real = np.zeros(num_complex_comp)
total_amp_prot_wat_dens_corr_imag = np.zeros(num_complex_comp)

if (int(params['solv_dens_corr'])):
    bins_per_sec_solv_dens_corr = int(50)
    num_sec_solv_dens_corr = int((num_cos_thet + 1)*num_phi)
    rad_dens_corr = np.zeros(num_sec_solv_dens_corr)
    count_secs = 0

    for i in range(num_cos_thet + 1):
        for j in range(num_phi):
      
            if (j == 0):
                phi_prev = num_phi - 1
            else:
                phi_prev = j - 1

            if (i == 0):
                rad_val_1 = rads[0]
                rad_val_2 = rads[int(1+ i*num_phi + phi_prev)]
                rad_val_3 = rads[int(1+ i*num_phi + j)]
                rad_dens_corr[count_secs] = np.max(np.array([rad_val_1, rad_val_2, rad_val_3]))
                count_secs = count_secs + 1

            elif (i == num_cos_thet):
                rad_val_1 = rads[num_points_1 - 1]
                rad_val_2 = rads[int(1+ (i-1)*num_phi + phi_prev)]
                rad_val_3 = rads[int(1+ (i-1)*num_phi + j)]
                rad_dens_corr[count_secs] = np.max(np.array([rad_val_1, rad_val_2, rad_val_3]))
                count_secs = count_secs + 1

            else:
                rad_val_1 = rads[int(1 + (i-1)*num_phi + phi_prev)]
                rad_val_2 = rads[int(1 + (i-1)*num_phi + j)]
                rad_val_3 = rads[int(1 + i*num_phi + phi_prev)]
                rad_val_4 = rads[int(1 + i*num_phi + j)]
                rad_val_5 = rads[int(num_points_1 + (i-1)*num_phi + phi_prev)]
                max_rad   = np.max(np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4]))
                min_rad   = np.min(np.array([rad_val_1, rad_val_2, rad_val_3, rad_val_4]))
            
                if (rad_val_5 >= max_rad):
                    rad_dens_corr[count_secs] = max_rad
                elif (rad_val_5 <= min_rad):
                    rad_dens_corr[count_secs] = min_rad
                else:
                    rad_dens_corr[count_secs] = (max_rad + min_rad)/2.0

                count_secs = count_secs + 1
 
    rad_bins_dens_corr  = np.zeros([num_sec_solv_dens_corr, bins_per_sec_solv_dens_corr])
    vol_bins_dens_corr  = np.zeros([num_sec_solv_dens_corr, bins_per_sec_solv_dens_corr])
    coor_bins_dens_corr = np.zeros([num_sec_solv_dens_corr, bins_per_sec_solv_dens_corr, 3]) 

    for i in range(num_cos_thet + 1):

        if (i == 0):
            cos_thet_loop = -1 + cos_thet_bin/4.0
            
        elif (i == num_cos_thet):
            cos_thet_loop = -1 + cos_thet_bin*(num_cos_thet - 0.25)

        else:
            cos_thet_loop = -1 + (cos_thet_bin)*i

        for j in range(num_phi):
            num_sec = int(i*num_phi + j)
            phi_loop = phi_bin*(j) - np.pi
    
            for k in range(bins_per_sec_solv_dens_corr):
                rad_bins_dens_corr = rad_dens_corr[num_sec]*(((2*k+1)/(2*bins_per_sec_solv_dens_corr))**(1/3))
                coor_bins_dens_corr[num_sec, k, 0] = rad_bins_dens_corr * np.sin(np.arccos(cos_thet_loop)) * np.cos(phi_loop)
                coor_bins_dens_corr[num_sec, k, 1] = rad_bins_dens_corr * np.sin(np.arccos(cos_thet_loop)) * np.sin(phi_loop)
                coor_bins_dens_corr[num_sec, k, 2] = rad_bins_dens_corr * (cos_thet_loop)

                if (i == 0):
                    vol_bins_dens_corr[num_sec, k] = ((rad_dens_corr[num_sec]**3.0/(bins_per_sec_solv_dens_corr))*(cos_thet_bin)*(phi_bin))/6.0
            
                elif (i == num_cos_thet):
                    vol_bins_dens_corr[num_sec, k] = ((rad_dens_corr[num_sec]**3.0/(bins_per_sec_solv_dens_corr))*(cos_thet_bin)*(phi_bin))/6.0
            
                else:
                    vol_bins_dens_corr[num_sec, k] = ((rad_dens_corr[num_sec]**3.0/(bins_per_sec_solv_dens_corr))*(cos_thet_bin)*(phi_bin))/3.0

    total_vol_bins = np.sum(vol_bins_dens_corr)

    for frame_bloc in range(floor_frames):
        if __name__ == '__main__':
            with multiprocessing.Pool() as pool:
                count_solv_dens_loop = pool.map(solv_dens_corr_prot_solv, range(int(frame_bloc*num_frames_once), int((frame_bloc + 1)*num_frames_once)))
                if (frame_bloc == 0):
                    count_solv_dens = count_solv_dens_loop
                else:
                    count_solv_dens = count_solv_dens + count_solv_dens_loop

    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            count_solv_dens_loop = pool.map(solv_dens_corr_prot_solv, range(int(floor_frames*num_frames_once), int(nframes)))
            if (floor_frames == 0):
                count_solv_dens = count_solv_dens_loop
            else:
                count_solv_dens = count_solv_dens + count_solv_dens_loop

    count_solv_dens_one       = np.zeros([num_sec_solv_dens_corr, bins_per_sec_solv_dens_corr])
    count_solv_dens_total     = np.zeros([num_sec_solv_dens_corr, bins_per_sec_solv_dens_corr])
    dens_bins_all             = np.zeros([num_sec_solv_dens_corr, bins_per_sec_solv_dens_corr])
    bulk_dens_sim             = 0.0

    for i in range(nframes):
        first, *rest            = count_solv_dens
        count_solv_dens_one     = first
        total_waters_frame      = np.sum(count_solv_dens_one)
        waters_frame_bulk       = (nOxygens_water - total_waters_frame)
        bulk_dens_sim += (waters_frame_bulk*10.0)/((ucelllengths[i, 0]*ucelllengths[i, 1]*ucelllengths[i,2] - total_vol_bins)*1000)
        count_solv_dens_total   = count_solv_dens_total + count_solv_dens_one  
        count_solv_dens         = rest

    count_solv_dens_total = count_solv_dens_total/nframes
    bulk_dens_sim = bulk_dens_sim/nframes
    print('Bulk density of waters [e/A^3] in the protein-water simulation is:%.6f' %(bulk_dens_sim))
    dens_bins_all = np.divide(count_solv_dens_total, vol_bins_dens_corr)
    dens_bins_all = dens_bins_all/100.0
    dens_corr = (params['corr_solv_dens'] - bulk_dens_sim)
    dens_corr_all = dens_bins_all * (dens_corr/bulk_dens_sim)
    number_corr_all = np.multiply(dens_corr_all, vol_bins_dens_corr*1000.0)

    floor_secs = np.floor_divide(num_sec_solv_dens_corr, num_frames_once)

    for sec_bloc in range(floor_secs):
        if __name__ == '__main__':
            with multiprocessing.Pool() as pool:
                prot_wat_dens_corr_amps_loop = pool.map(prot_wat_dens_corr_scattering, range(int(sec_bloc*num_frames_once), int((sec_bloc + 1)*num_frames_once)))
                if (sec_bloc == 0):
                    prot_wat_dens_corr_amps = prot_wat_dens_corr_amps_loop
                else:
                    prot_wat_dens_corr_amps = prot_wat_dens_corr_amps + prot_wat_dens_corr_amps_loop

    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            prot_wat_dens_corr_amps_loop = pool.map(prot_wat_dens_corr_scattering, range(int(floor_secs*num_frames_once), int(num_sec_solv_dens_corr)))
            if (floor_secs == 0):
                prot_wat_dens_corr_amps = prot_wat_dens_corr_amps_loop
            else:
                prot_wat_dens_corr_amps = prot_wat_dens_corr_amps + prot_wat_dens_corr_amps_loop

    num_amp_entry = len(prot_wat_dens_corr_amps)

    for i in range(num_amp_entry):
        first, *rest            = prot_wat_dens_corr_amps
        sec_real, sec_imag      = first
        total_amp_prot_wat_dens_corr_real = total_amp_prot_wat_dens_corr_real + sec_real 
        total_amp_prot_wat_dens_corr_imag = total_amp_prot_wat_dens_corr_imag + sec_imag
        prot_wat_dens_corr_amps = rest  

    print('Bulk density of waters [e/A3] corrected to %.6f in the protein-water simulation' %(params['corr_solv_dens']))

for frame_bloc in range(floor_frames):
    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            prot_wat_amps_loop = pool.map(prot_wat_scattering, range(int(frame_bloc*num_frames_once), int((frame_bloc + 1)*num_frames_once)))
            if (frame_bloc == 0):
                prot_wat_amps = prot_wat_amps_loop
            else:
                prot_wat_amps = prot_wat_amps + prot_wat_amps_loop

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        prot_wat_amps_loop = pool.map(prot_wat_scattering, range(int(floor_frames*num_frames_once), int(nframes)))
        if (floor_frames == 0):
            prot_wat_amps = prot_wat_amps_loop
        else:
            prot_wat_amps = prot_wat_amps + prot_wat_amps_loop

num_amp_entry = len(prot_wat_amps)

total_amp_prot_wat_real = np.zeros(num_complex_comp)
total_amp_prot_wat_imag = np.zeros(num_complex_comp)
total_intens_prot_wat   = np.zeros(num_complex_comp)
avg_amp_prot_wat_real   = np.zeros(num_complex_comp)
avg_amp_prot_wat_imag   = np.zeros(num_complex_comp)
avg_intens_prot_wat     = np.zeros(num_complex_comp)

for i in range(num_amp_entry):
    first, *rest            = prot_wat_amps
    frame_real, frame_imag  = first
    frame_real              = frame_real + total_amp_prot_wat_dens_corr_real
    frame_imag              = frame_imag + total_amp_prot_wat_dens_corr_imag
    total_amp_prot_wat_real = total_amp_prot_wat_real + frame_real 
    total_amp_prot_wat_imag = total_amp_prot_wat_imag + frame_imag
    total_intens_prot_wat   = total_intens_prot_wat + np.multiply(frame_real, frame_real) + np.multiply(frame_imag, frame_imag) 
    prot_wat_amps           = rest

avg_amp_prot_wat_real = total_amp_prot_wat_real / num_amp_entry
avg_amp_prot_wat_imag = total_amp_prot_wat_imag / num_amp_entry
avg_intens_prot_wat   = total_intens_prot_wat / num_amp_entry 

print('Amplitudes calculated for the protein in solvent simulation system')

dcdfile      = input_files['traj_bulk_solv']
topfile      = input_files['topol_bulk_solv']
traj         = md.load(dcdfile, top = topfile)
topol        = traj.topology
nframes      = traj.n_frames
natoms       = traj.n_atoms
ucelllengths = traj.unitcell_lengths
ucellangles  = traj.unitcell_angles

num_frames_once = int(num_cores)
floor_frames = np.floor_divide(nframes, num_frames_once)

oxygens_water = [atom.index for atom in topol.atoms if (atom.residue.is_water and atom.name[0] == 'O')]
nOxygens_water = len(oxygens_water)

hydrogens_water = [atom.index for atom in topol.atoms if (atom.residue.is_water and atom.name[0] == 'H')]
nHydrogens_water = len(hydrogens_water)

print('Bulk solvent simulation trajectory loaded')

print('The number of waters in the bulk water simulation is:', nOxygens_water)

min_boxlength_x = np.min(ucelllengths[:, 0])
min_boxlength_y = np.min(ucelllengths[:, 1])
min_boxlength_z = np.min(ucelllengths[:, 2])
env_cross = False

for i in range(num_entries_env):
    if (np.absolute(sp_x[i]*rads[i]) >= min_boxlength_x/2):
        env_cross = True
        break

    if (np.absolute(sp_y[i]*rads[i]) >= min_boxlength_y/2):
        env_cross = True
        break   

    if (np.absolute(sp_z[i]*rads[i]) >= min_boxlength_z/2):
        env_cross = True
        break

if (env_cross):
    print('Warning! Dimensions of envelope excede the boundaries of the bulk solvent simulation box. Consider increasing the size of the simulation box')
else:
    print('Envelope dimensions within the boundaries of the bulk solvent simulation box')

xyz_reimaged   = np.zeros([nframes, natoms, 3]) 

if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
                reimaged_frames = pool.map(reimage_frames_bulk_solv, range(nframes))

num_reimaged_frames = len(reimaged_frames)

for i in range(num_reimaged_frames):
    first_frame, *rest_frame = reimaged_frames
    xyz_reimaged[i, :, :]    = first_frame
    reimaged_frames          = rest_frame

traj         = md.Trajectory(xyz = xyz_reimaged, topology = topol, unitcell_lengths = ucelllengths, unitcell_angles = ucellangles) 

del xyz_reimaged
del reimaged_frames

total_amp_bulk_wat_dens_corr_real = np.zeros(num_complex_comp)
total_amp_bulk_wat_dens_corr_imag = np.zeros(num_complex_comp)

if (int(params['solv_dens_corr'])):
    bulk_dens_sim = 0.0

    for i in range(nframes):
        bulk_dens_sim += (nOxygens_water*10.0)/(ucelllengths[i,0]*ucelllengths[i,1]*ucelllengths[i,2]*1000.0)

    bulk_dens_sim = bulk_dens_sim/nframes
    print('Density of waters [e/A^3] in the bulk-water simulation is:%.6f' %(bulk_dens_sim))
    dens_corr_all = params['corr_solv_dens'] - bulk_dens_sim
    number_corr_all = dens_corr_all*1000.0*vol_bins_dens_corr

    floor_secs = np.floor_divide(num_sec_solv_dens_corr, num_frames_once)

    for sec_bloc in range(floor_secs):
        if __name__ == '__main__':
            with multiprocessing.Pool() as pool:
                bulk_wat_dens_corr_amps_loop = pool.map(prot_wat_dens_corr_scattering, range(int(sec_bloc*num_frames_once), int((sec_bloc + 1)*num_frames_once)))
                if (sec_bloc == 0):
                    bulk_wat_dens_corr_amps = bulk_wat_dens_corr_amps_loop
                else:
                    bulk_wat_dens_corr_amps = bulk_wat_dens_corr_amps + bulk_wat_dens_corr_amps_loop

    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            bulk_wat_dens_corr_amps_loop = pool.map(prot_wat_dens_corr_scattering, range(int(floor_secs*num_frames_once), int(num_sec_solv_dens_corr)))
            if (floor_secs == 0):
                bulk_wat_dens_corr_amps = bulk_wat_dens_corr_amps_loop
            else:
                bulk_wat_dens_corr_amps = bulk_wat_dens_corr_amps + bulk_wat_dens_corr_amps_loop

    num_amp_entry = len(bulk_wat_dens_corr_amps)

    for i in range(num_amp_entry):
        first, *rest            = bulk_wat_dens_corr_amps
        sec_real, sec_imag      = first
        total_amp_bulk_wat_dens_corr_real = total_amp_bulk_wat_dens_corr_real + sec_real 
        total_amp_bulk_wat_dens_corr_imag = total_amp_bulk_wat_dens_corr_imag + sec_imag
        bulk_wat_dens_corr_amps = rest  

    print('Density of waters [e/A3] corrected to %.6f in the bulk-water simulation' %(params['corr_solv_dens']))

for frame_bloc in range(floor_frames):
    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            bulk_wat_amps_loop = pool.map(bulk_wat_scattering, range(int(frame_bloc*num_frames_once), int((frame_bloc + 1)*num_frames_once)))
            if (frame_bloc == 0):
                bulk_wat_amps = bulk_wat_amps_loop
            else:
                bulk_wat_amps = bulk_wat_amps + bulk_wat_amps_loop

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        bulk_wat_amps_loop = pool.map(bulk_wat_scattering, range(int(floor_frames*num_frames_once), int(nframes)))
        if (floor_frames == 0):
            bulk_wat_amps = bulk_wat_amps_loop
        else:
            bulk_wat_amps = bulk_wat_amps + bulk_wat_amps_loop

num_bulk_amp_entry = len(bulk_wat_amps)

total_amp_bulk_wat_real = np.zeros(num_complex_comp)
total_amp_bulk_wat_imag = np.zeros(num_complex_comp)
total_intens_bulk_wat   = np.zeros(num_complex_comp)
avg_amp_bulk_wat_real   = np.zeros(num_complex_comp)
avg_amp_bulk_wat_imag   = np.zeros(num_complex_comp)
avg_intens_bulk_wat     = np.zeros(num_complex_comp)

for i in range(num_bulk_amp_entry):
    first_bulk, *rest_bulk            = bulk_wat_amps
    frame_bulk_real, frame_bulk_imag  = first_bulk
    frame_bulk_real                   = frame_bulk_real + total_amp_bulk_wat_dens_corr_real
    frame_bulk_imag                   = frame_bulk_imag + total_amp_bulk_wat_dens_corr_imag
    total_amp_bulk_wat_real = total_amp_bulk_wat_real + frame_bulk_real 
    total_amp_bulk_wat_imag = total_amp_bulk_wat_imag + frame_bulk_imag
    total_intens_bulk_wat   = total_intens_bulk_wat + np.multiply(frame_bulk_real, frame_bulk_real) + np.multiply(frame_bulk_imag, frame_bulk_imag) 
    bulk_wat_amps           = rest_bulk

avg_amp_bulk_wat_real = total_amp_bulk_wat_real / num_bulk_amp_entry
avg_amp_bulk_wat_imag = total_amp_bulk_wat_imag / num_bulk_amp_entry
avg_intens_bulk_wat   = total_intens_bulk_wat / num_bulk_amp_entry 

print('Amplitudes calculated for the bulk solvent simulation system')

num_entries_bulk = len(avg_amp_bulk_wat_real)

num_sphere = params['num_orient']
quan_int                     = np.zeros(num_complex_comp, float)
orient_avg                   = np.zeros(num_q_mag, float)
block_fac                    = 1

for i in range(num_complex_comp):
    quan_int[i] = (avg_amp_prot_wat_real[i] - avg_amp_bulk_wat_real[i])**2.0 + (avg_amp_prot_wat_imag[i] - avg_amp_bulk_wat_imag[i])**2.0 + (avg_intens_prot_wat[i] - (avg_amp_prot_wat_real[i]**2.0 + avg_amp_prot_wat_imag[i]**2.0)) - block_fac*(avg_intens_bulk_wat[i] - (avg_amp_bulk_wat_real[i]**2.0 + avg_amp_bulk_wat_imag[i]**2.0))

orient_avg[0] = quan_int[0]

for i in range(1, num_q_mag):
    orient_avg[i] = np.sum(quan_int[1 + (i-1)*num_sphere : 1 + i*num_sphere])/num_sphere

fmt_5 = "%15.8f %20.10f\n"
out_5 = []

if (int(params['fit_exp_data'])):
    for i in range(num_exp_data):
        a5 = fmt_5 % (q_mag[i+q_exp_start]/10.0, orient_avg[i+q_exp_start])
        out_5.append(a5)
else:
    for i in range(int(params['num_q'])):
        a5 = fmt_5 % (q_mag[i]/10.0, orient_avg[i])
        out_5.append(a5)

open(output_files['comp_iq'], 'w').writelines(out_5)

print('Simulated background subtracted intensities calculated for the system and written to the specified output file')

if (int(params['fit_exp_data'])):
    if (int(params['num_fit_para']) == 1):
        
        tol = 0.000001
        max_iter = 10000
        f_iter = iq_exp[0]/orient_avg[int(q_exp_start)]

        for j in range(max_iter):

            func = 0.0
            func_der = 0.0

            for k in range(num_exp_data):
                func = func + (2*f_iter*orient_avg[q_exp_start + k]**2.0 - 2*orient_avg[q_exp_start + k]*iq_exp[k])/err_exp[k]**2.0
                func_der = func_der + (2*orient_avg[q_exp_start + k]**2.0)/err_exp[k]**2.0

            func = func/num_exp_data
            func_der = func_der/num_exp_data

            f_new = f_iter - func/func_der

            if np.absolute(f_iter - f_new) <= tol:
                break

            f_iter = f_new

        orient_avg_scale = orient_avg*f_new

        X2 = 0.0

        for k in range(num_exp_data):
            X2 = X2 + ((orient_avg_scale[q_exp_start+k] - iq_exp[k])/(err_exp[k]))**2.0

        X2 = X2/num_exp_data

        with open(output_files['comp_iq_exp_fit'], 'w') as text_file:
            text_file.write('X^2 for the computed profile with respect to the experimental profile: %.6f \n' %(X2))

        with open(output_files['comp_iq_exp_fit'], 'a') as text_file:
            text_file.write('q \t I(q) \n')            

        fmt = "%20.10f %20.10f\n"
        out = []

        for i in range(num_exp_data):
            a = fmt % (q_mag[i+q_exp_start]/10.0, orient_avg_scale[i+q_exp_start])
            out.append(a)

        open(output_files['comp_iq_exp_fit'], 'a').writelines(out)

    elif (int(params['num_fit_para']) == 2):
        max_iter                             = 10000
        max_ini_change                       = 100
        max_c_ini_per                        = 5
        tol_f_rel                            = 0.0001
        tol_c_rel                            = 0.0001
        search_para                          = False

        for main_iter in range(max_ini_change):

            mod_ini_change = int(num_exp_data/max_ini_change)
            f_ini          = iq_exp[main_iter*mod_ini_change]/orient_avg[int(main_iter*mod_ini_change + q_exp_start)]
            c_ini          = -1.0**main_iter*(iq_exp[0]*max_c_ini_per*main_iter)/(100*max_ini_change)
            para_two_ini   = np.array([[f_ini], [c_ini]])

            for i in range(max_iter):

                sum_g      = 0.0
                sum_h      = 0.0
                sum_g_derf = 0.0
                sum_g_derc = 0.0
                sum_h_derf = 0.0
                sum_h_derc = 0.0

                for j in range(num_exp_data):

                    sum_g      = sum_g + (2*f_ini*orient_avg[q_exp_start + j]**2.0 + 2*c_ini*orient_avg[q_exp_start + j] - 2*orient_avg[q_exp_start + j]*iq_exp[j])/(err_exp[j]**2.0)
                    sum_h      = sum_h + (2*c_ini + 2*f_ini*orient_avg[q_exp_start + j] - 2*iq_exp[j])/(err_exp[j]**2.0)
                    sum_g_derf = sum_g_derf + (2*orient_avg[q_exp_start+j]**2.0)/(err_exp[j]**2.0)
                    sum_g_derc = sum_g_derc + (2*orient_avg[q_exp_start+j])/(err_exp[j]**2.0)
                    sum_h_derf = sum_h_derf + (2*orient_avg[q_exp_start+j])/(err_exp[j]**2.0)
                    sum_h_derc = sum_h_derc + 2/(err_exp[j]**2.0)

                sum_g      = sum_g/num_exp_data
                sum_h      = sum_h/num_exp_data
                sum_g_derf = sum_g_derf/num_exp_data
                sum_g_derc = sum_g_derc/num_exp_data
                sum_h_derf = sum_h_derf/num_exp_data
                sum_h_derc = sum_h_derc/num_exp_data

                jac_mat     = np.array([[sum_g_derf, sum_g_derc], [sum_h_derf, sum_h_derc]])
                det_jac_mat = np.linalg.det(jac_mat)
                if (det_jac_mat == 0):
                    break

                func_mat    = np.array([[sum_g], [sum_h]])

                para_two_new = para_two_ini - np.matmul(np.linalg.inv(jac_mat), func_mat)
                f_new = para_two_new[0, 0]
                c_new = para_two_new[1, 0]

                if ((np.abs((f_new - f_ini)*orient_avg[q_exp_start]/iq_exp[0]) < tol_f_rel) and (np.abs((c_new - c_ini)/iq_exp[0]) < tol_c_rel)):
                    search_para = True
                    break

                para_two_ini = para_two_new
                f_ini = f_new
                c_ini = c_new

            if (search_para):
                break

        orient_avg_scale = orient_avg*f_new + c_new

        X2 = 0.0

        for k in range(num_exp_data):
            X2 = X2 + ((orient_avg_scale[q_exp_start+k] - iq_exp[k])/(err_exp[k]))**2.0

        X2 = X2/num_exp_data

        with open(output_files['comp_iq_exp_fit'], 'w') as text_file:
            text_file.write('X^2 for the computed profile with respect to the experimental profile: %.6f \n' %(X2))

        with open(output_files['comp_iq_exp_fit'], 'a') as text_file:
            text_file.write('q \t I(q) \n')            

        fmt = "%20.10f %20.10f\n"
        out = []

        for i in range(num_exp_data):
            a = fmt % (q_mag[i+q_exp_start]/10.0, orient_avg_scale[i+q_exp_start])
            out.append(a)

        open(output_files['comp_iq_exp_fit'], 'a').writelines(out)

if (int(params['guinier_analysis'])):
    if (int(params['fit_exp_data'])):
        if (int(params['num_fit_para']) == 0):
            exp_iq_rel     = orient_avg[-10:]/orient_avg[0]
            log_exp_iq_rel = np.log(exp_iq_rel)
            exp_q2         = np.multiply(q_mag[-10:], q_mag[-10:])
            m_exp,c_exp    = np.polyfit(exp_q2, log_exp_iq_rel, 1)
            Rg_guinier     = np.sqrt(-3*m_exp)

            with open(output_files['guinier_file'], 'w') as text_file:
                text_file.write('Guinier analysis based radius of gyration computed from the scattering intensities in Angstroms: %.6f \n' %(Rg_guinier*10.0))

            with open(output_files['guinier_file'], 'a') as text_file:
                text_file.write('Intensity at 0 q (I(0)) for the system: %f \n' %(orient_avg[0]))
        
            with open(output_files['guinier_file'], 'a') as text_file:
                text_file.write('q_guinier \t I(q)_guinier \n')

            fmt = "%15.8f %20.10f\n"
            out = []

            a = fmt % (q_mag[0]/10.0, orient_avg[0])
            out.append(a)

            for i in range(10):
                a = fmt % (q_mag[num_q_mag -10 + i]/10.0, orient_avg[num_q_mag -10 + i])
                out.append(a)
            
            open(output_files['guinier_file'], 'a').writelines(out)

        elif (int(params['num_fit_para']) == 1):
            exp_iq_rel     = orient_avg_scale[-10:]/orient_avg_scale[0]
            log_exp_iq_rel = np.log(exp_iq_rel)
            exp_q2         = np.multiply(q_mag[-10:], q_mag[-10:])
            m_exp,c_exp    = np.polyfit(exp_q2, log_exp_iq_rel, 1)
            Rg_guinier     = np.sqrt(-3*m_exp)

            with open(output_files['guinier_file'], 'w') as text_file:
                text_file.write('Guinier analysis based radius of gyration computed from the scattering intensities in Angstroms: %.6f \n' %(Rg_guinier*10))

            with open(output_files['guinier_file'], 'a') as text_file:
                text_file.write('Intensity at 0 q (I(0)) for the system: %f \n' %(orient_avg_scale[0]))
        
            with open(output_files['guinier_file'], 'a') as text_file:
                text_file.write('q_guinier \t I(q)_guinier \n')

            fmt = "%15.8f %20.10f\n"
            out = []

            a = fmt % (q_mag[0]/10.0, orient_avg_scale[0])
            out.append(a)

            for i in range(10):
                a = fmt % (q_mag[num_q_mag -10 + i]/10.0, orient_avg_scale[num_q_mag -10 + i])
                out.append(a)
            
            open(output_files['guinier_file'], 'a').writelines(out)

        elif (int(params['num_fit_para']) == 2):
            exp_iq_rel     = orient_avg_scale[-10:]/orient_avg_scale[0]
            log_exp_iq_rel = np.log(exp_iq_rel)
            exp_q2         = np.multiply(q_mag[-10:], q_mag[-10:])
            m_exp,c_exp    = np.polyfit(exp_q2, log_exp_iq_rel, 1)
            Rg_guinier     = np.sqrt(-3*m_exp)

            with open(output_files['guinier_file'], 'w') as text_file:
                text_file.write('Guinier analysis based radius of gyration computed from the scattering intensities in Angstroms: %.6f \n' %(Rg_guinier*10.0))

            with open(output_files['guinier_file'], 'a') as text_file:
                text_file.write('Intensity at 0 q (I(0)) for the system: %f \n' %(orient_avg_scale[0]))
        
            with open(output_files['guinier_file'], 'a') as text_file:
                text_file.write('q_guinier \t I(q)_guinier \n')

            fmt = "%15.8f %20.10f\n"
            out = []

            a = fmt % (q_mag[0]/10.0, orient_avg_scale[0])
            out.append(a)

            for i in range(10):
                a = fmt % (q_mag[num_q_mag -10 + i]/10.0, orient_avg_scale[num_q_mag -10 + i])
                out.append(a)
            
            open(output_files['guinier_file'], 'a').writelines(out)

    else:
        exp_iq_rel     = orient_avg[-10:]/orient_avg[0]
        log_exp_iq_rel = np.log(exp_iq_rel)
        exp_q2         = np.multiply(q_mag[-10:], q_mag[-10:])
        m_exp,c_exp    = np.polyfit(exp_q2, log_exp_iq_rel, 1)
        Rg_guinier     = np.sqrt(-3*m_exp)

        with open(output_files['guinier_file'], 'w') as text_file:
            text_file.write('Guinier analysis based radius of gyration computed from the scattering intensities in Angstroms: %.6f \n' %(Rg_guinier*10.0))

        with open(output_files['guinier_file'], 'a') as text_file:
            text_file.write('Intensity at 0 q (I(0)) for the system: %f \n' %(orient_avg[0]))
        
        with open(output_files['guinier_file'], 'a') as text_file:
            text_file.write('q_guinier \t I(q)_guinier \n')

        fmt = "%15.8f %20.10f\n"
        out = []

        a = fmt % (q_mag[0]/10.0, orient_avg[0])
        out.append(a)

        for i in range(10):
            a = fmt % (q_mag[num_q_mag -10 + i]/10.0, orient_avg[num_q_mag -10 + i])
            out.append(a)
            
        open(output_files['guinier_file'], 'a').writelines(out)

end_time = time.time()

print('All computations done!')
print('Total time required for scattering profile computation in hours:', (end_time - start_time)/3600.0)
