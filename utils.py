#!/usr/bin/env python3
import glob
import os
import re
import shutil

import ase
import ase.io.castep
import ase.optimize
import matplotlib.pyplot as plt
import numpy as np
from ase import io
from ase.calculators.castep import Castep
from ase.calculators.emt import EMT
from ase.calculators.nwchem import NWChem
from ase.calculators.onetep import Onetep
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.io import read
from ase.io import write
from ase.neb import NEBTools
from ase.optimize import BFGS
from ase.optimize import (FIRE, LBFGS, LBFGSLineSearch, BFGSLineSearch,
                          GPMin)
from ase.optimize.precon import Exp, PreconLBFGS
from ase.visualize import view
from ase.visualize.plot import plot_atoms


def reduced_mass(mass, reac_co, poss_co, acc=4):
    """

    :param mass:
    :param reac_co:
    :param poss_co:
    :param acc:
    :return:
    """
    from findiff import FinDiff
    # Inverse mass
    mass = np.power(mass, -1)
    poss_co = np.array(poss_co)
    msum = np.zeros(np.shape(mass)[-1])
    plt.close()
    # Loop over elemental masses
    for i in range(np.shape(mass)[-1]):
        # Define the partial derivative
        r = np.linalg.norm(poss_co[:, i, :] - poss_co[0, i, :], axis=-1)  # -poss_co[0,i,:]
        d_dx = FinDiff(0, r, 1, acc=acc)

        # Apply the derivative
        pdiv = d_dx(np.array(reac_co))

        plt.figure(1)
        plt.plot(reac_co, r)
        plt.figure(2)
        plt.plot(reac_co, pdiv)

        pdiv_norm = np.linalg.norm(pdiv)
        print(pdiv_norm)
        # determine sum elements
        msum[i] = mass[0, i] * np.square(np.abs(pdiv_norm))
    # Determine the full sum
    plt.figure(1)
    n_plot(r'Path $\AA$', 'Image vector, r')
    plt.savefig('path_vs_r.pdf')
    plt.close()
    plt.figure(2)
    n_plot(r'Path $\AA$', r'$\frac{\partial q}{\partial r}$')
    plt.savefig('path_vs_dq_dr.pdf')
    plt.close()
    mu = 1.0 / np.sum(msum)
    return mu


def new_reduced_mass(mass, reac_co, poss_co, acc=4):
    from findiff import FinDiff
    reac_co = np.array(reac_co)
    poss_co = np.array(poss_co)
    num_atoms = len(mass)
    num_images = len(reac_co)
    print(reac_co)

    r_arr = np.zeros([num_images, num_atoms])
    r_arr_n = np.zeros([num_images, num_atoms])
    for i in range(num_atoms):
        r_arr[:, i] = np.linalg.norm(poss_co[:, i, :], axis=-1)  # - poss_co[0, i, :]
        r_arr_n[:, i] = np.linalg.norm(poss_co[:, i, :] - poss_co[0, i, :], axis=-1)

    for i in range(num_atoms):
        plt.plot(reac_co, r_arr[:, i], label='%s%i' % (E[i], i))
    plt.legend(loc='best')
    n_plot(r'Reaction path, s [$\AA$]', 'Image coordinate vector, r [$\AA$]')
    plt.savefig('path_vs_r.pdf')
    plt.show()

    for i in range(num_atoms):
        plt.plot(reac_co, r_arr_n[:, i], label='%s%i' % (E[i], i))
    plt.legend(loc='best')
    n_plot(r'Reaction path, s [$\AA$]', 'Normalised image coordinate vector, r [$\AA$]')
    plt.savefig('path_vs_r_norm.pdf')
    plt.show()

    rm = np.zeros([num_atoms, num_images])
    for i in range(num_atoms):
        x_i = r_arr[:, i]
        d_dx = FinDiff(0, x_i, 1, acc=acc)
        pdiv = d_dx(reac_co)
        rm[i, :] = 1 / mass[i] * np.dot(pdiv, pdiv)

    mu = 1 / np.sum(rm, axis=0)
    plt.plot(reac_co, mu)
    n_plot(r'Reaction path, s [$\AA$]', 'reduced mass, $\mu$ [amu]')
    plt.savefig('path_vs_mu.pdf')
    plt.show()
    return mu


# Performs NEB analysis
def neb_anal_cont(nebtools, name):
    # Plot the NEB diagram
    try:
        # Get the calculated barrier and the energy change of the reaction.
        Ef, dE = nebtools.get_barrier()
        print('Calculated barrier and the energy change:', Ef, dE)

        # Get the barrier without any interpolation between highest images.
        Ef, dE = nebtools.get_barrier(fit=False)
        print('(w/o any interpolation) Calculated barrier and the energy change:', Ef, dE)

        # Get the actual maximum force at this point in the simulation.
        max_force = nebtools.get_fmax()
        print('Max force:', max_force)

        # Create a figure with custom parameters.
        fig = plt.figure(figsize=(5.5, 4.0))
        ax = fig.add_axes((0.15, 0.15, 0.8, 0.75))
        nebtools.plot_band(ax)
        fig.savefig('NEB.pdf')
        os_plot_show()
    except:
        print('problem')

    try:
        # Grab the fitting parameters
        fitting = nebtools.get_fit()  # s, E, Sfit, Efit, lines
        s = fitting[0]
        E = fitting[1]
        Sfit = fitting[2]
        Efit = fitting[3]
        lines = fitting[4]
        plt.plot(Sfit, Efit)
        plt.savefig('Sfit_vs_Efit.pdf')
        os_plot_show()

        plt.scatter(s, E)
        plt.savefig('s_vs_E.pdf')
        os_plot_show()

        # Grab the images
        images = nebtools._images
        R = [atoms.positions for atoms in images]
        # Need to work on an animation
        mass = [atoms.get_masses() for atoms in images]
        print('R shape:', np.shape(R))
        print('mass shape:', np.shape(mass))
        print('mass matrix:\n', mass)
        mu = reduced_mass(mass, s, R, acc=4)
        print('Reduced mass:', mu)

    except:
        print('problem')

    # Plot the convergence
    try:
        files = sub_file_list(os.getcwd(), '.log')
        file = [i for i in files if name in i]
        for ii, val in enumerate(file):
            A = np.genfromtxt(val, skip_header=1, usecols=(1, 2, 3, 4))
            plt.plot(A[:, 0], A[:, -1], marker='o')
            n_plot('Iteration number', 'Max force')
            plt.savefig('%s__N_vs_dF.pdf' % val)
            os_plot_show()
    except:
        print('Plot the convergence problem')
    return None


# Performs NEB analysis
def neb_anal(N_images, title='NEB'):
    try:
        # Grab the files
        files = sub_file_list(os.getcwd(), '.traj')
        file = 'a2b.traj'
        # Read in the images
        images = read(file + '@-' + str(N_images) + ':')
        # Grab the tool box
        nebtools = NEBTools(images)
        # Get the calculated barrier and the energy change of the reaction.
        Ef, dE = nebtools.get_barrier()
        print('Calculated barrier and the energy change:', Ef, dE)

        # Get the barrier without any interpolation between highest images.
        Ef, dE = nebtools.get_barrier(fit=False)
        print('(w/o any interpolation) Calculated barrier and the energy change:', Ef, dE)

        # Get the actual maximum force at this point in the simulation.
        max_force = nebtools.get_fmax()
        print('Max force:', max_force)

        # Create a figure with custom parameters.
        fig = plt.figure(figsize=(5.5, 4.0))
        ax = fig.add_axes((0.15, 0.15, 0.8, 0.75))
        nebtools.plot_band(ax)
        fig.savefig('NEB.pdf')
        os_plot_show()
    except:
        print('problem')

    # Grab the fitting parameters
    fitting = nebtools.get_fit()  # s, E, Sfit, Efit, lines
    s = fitting[0]
    E = fitting[1]
    Sfit = fitting[2]
    Efit = fitting[3]
    lines = fitting[4]
    plt.plot(Sfit, Efit)
    plt.savefig('test.pdf')
    os_plot_show()

    plt.scatter(s, E)
    plt.savefig('test1.pdf')
    os_plot_show()

    # Grab the images
    images = nebtools._images
    R = [atoms.positions for atoms in images]
    mass = [atoms.get_masses() for atoms in images]
    print('R shape:', np.shape(R))
    print('mass shape:', np.shape(mass))
    print(mass)
    mu = reduced_mass(mass, s, R, acc=4)
    print('Reduced mass:', mu)

    # Plot the convergence
    files = sub_file_list(os.getcwd(), '.log')
    file = 'a2b.log'
    A = np.genfromtxt(file, skip_header=1, usecols=(1, 2, 3, 4))
    plt.plot(A[:, 0], A[:, -1], marker='o')
    plt.title(title)
    n_plot('Iteration number', 'Max force')
    plt.savefig('N_vs_dF.pdf')
    os_plot_show()
    return None


def gen_names(xc_list, gold_list, ele_list):
    f_type = ''  # xyz
    """
    xc_list = ['B3LYP']
    xc_list = ['B3LYP', 'PBE0']
    gold_list = ['gold', 'gold_impsol']

    ele_list = ['GC']
    ele_list = ['AT']

    ele_list = ['A', 'T', 'G', 'C']
    ele_list = ['GC']
    """
    # post_fix = '_fix_align'
    post_fix = '_ase'
    names_list = []
    for e in ele_list:  # Loop over elements
        for gold in gold_list:  # Loop over standards
            for xc in xc_list:  # loop over exchange correlation functionals
                # Fix the PBE0 case
                if xc.upper() == 'PBE0':
                    a = gold.split('_')
                    if len(a) == 1:
                        gold = a[0] + '_mbd'
                    else:
                        gold = ''
                        for i, val in enumerate(a):
                            if i == 1:
                                gold += 'mbd' + '_'
                            gold += val + '_'

                        # strip the trailing _
                        gold = gold[:-1]

                rect_xyz_file = r'%s_%s_%s%s%s' % (e.upper(), xc.upper(), gold.upper(), post_fix, f_type)
                prod_xyz_file = r'%s_taut_%s_%s%s%s' % (e.upper(), xc.upper(), gold.upper(), post_fix, f_type)
                print(rect_xyz_file)
                print(prod_xyz_file)
                names_list.append(rect_xyz_file)
                names_list.append(prod_xyz_file)
    return names_list


def bodge():
    xyz_folder = r'/users/ls00338/xyz/Gold'
    file = 'bodge_list.txt'
    reac = 'react.traj'
    prod = 'prod.traj'
    # load the files
    f_names = np.genfromtxt(file, dtype=str, delimiter='\n')
    f_names = gen_names(['B3LYP', 'PBE0'], ['gold', 'gold_impsol'], ['GC']) + gen_names(['B3LYP'],
                                                                                        ['gold', 'gold_impsol'],
                                                                                        ['AT']) + gen_names(
        ['B3LYP', 'PBE0'], ['gold', 'gold_impsol'], ['A', 'T', 'G', 'C'])
    # Make the list of work dirs
    work_dirs = ['W00299', 'W00300', 'W00305', 'W00306', 'W00307', 'W00308']
    work_dirs = [os.path.join(os.getcwd(), i) for i in work_dirs]

    # make the list of job dirs
    a = []
    for i in work_dirs:
        job_dirs = folder_list(i)
        job_dirs.sort(key=natural_keys)
        for j in job_dirs:
            a.append(os.path.join(os.path.join(i, j), reac))
            a.append(os.path.join(os.path.join(i, j), prod))
    assert len(f_names) == len(a)
    num = 0
    for i in range(len(f_names)):

        target = os.path.join(xyz_folder, f_names[i] + '.traj')
        if os.path.exists(a[i]) == True and int(os.path.getsize(a[i])) > 0:
            num += 1
            print(a[i])
            print(target)
            shutil.copy(a[i], target)
    print(num)
    return None


def harm_correc(g_omega, omega, T):
    """
    Determines the harmonic free energy correction.
    From slide 9/20 of:
    http://www.tcm.phy.cam.ac.uk/castep/CASTEP_talks_06/refson2.pdf
    :param g_omega: DOS
    :param omega: Phonon eigenvalues
    :param T: Temperature of the system
    :return: The correction X, in the free energy equation F = E + X
    """
    from scipy.integrate import simps
    h_bar = 6.582119569e-16  # eV s https://en.wikipedia.org/wiki/Planck_constant
    k_b = 8.617333262145e-5  # eV k^-1 https://en.wikipedia.org/wiki/Boltzmann_constant
    beta = 1 / (k_b * T)
    internal = g_omega * np.log(2 * np.sinh(0.5 * beta * h_bar * omega))
    return 1 / beta * simps(internal, omega)


#### autoNEB/NEB functions ####
# Gold standard params
def cas_in_prep(c_ob, xc_func, settings='gold', f_flush=False):
    """

    :param c_ob:
    :param xc_func:
    :param set: expected: gold, dev, mbd, impsol
    :return:
    """
    set_list = settings.split('_')
    c_ob.param.xc_functional = xc_func.upper()
    if 'gold' in set_list:
        print('Using gold settings', flush=f_flush)
        # Set the param file keywords
        c_ob.param.cut_off_energy = 1100
        c_ob.param.elec_energy_tol = 1.0E-8
        c_ob.param.elec_eigenvalue_tol = 1.0E-8
        c_ob.param.fine_grid_scale = 2.0

        c_ob.param.geom_energy_tol = 1.0E-6
        c_ob.param.geom_force_tol = 0.01
        c_ob.param.geom_stress_tol = 0.02
        c_ob.param.geom_disp_tol = 5.0E-4
        c_ob.param.geom_max_iter = 500
    elif 'extreme' in set_list:
        print('Using extreme settings', flush=f_flush)
        # Set the param file keywords
        c_ob.param.elec_method = 'EDFT'
        c_ob.param.cut_off_energy = 1500
        c_ob.param.elec_energy_tol = 5.0E-7
        c_ob.param.elec_eigenvalue_tol = 5.0E-7
        c_ob.param.fine_grid_scale = 2.0
    else:
        print('Warning not in settings...', flush=f_flush)
        c_ob.param.cut_off_energy = float(settings)

    c_ob.param.num_dump_cycles = 0
    c_ob.param.fix_occupancy = True
    c_ob.param.max_scf_cycles = 300
    c_ob.param.mixing_scheme = 'pulay'
    c_ob.param.opt_strategy = "speed"
    c_ob.param.reuse = True
    # Check to see if settings are given
    if 'dev' in set_list:
        c_ob.param.devel_code = "improve_wvfn"
    if 'mbd' in set_list:
        c_ob.param.sedc_apply = True
        c_ob.param.sedc_scheme = 'MBD'
    if 'impsol' in set_list:
        c_ob.param.implicit_solvent_apolar_term = True

    # Set the cell file keywords
    c_ob.cell.symmetry_generate = True
    c_ob.cell.fix_com = False
    c_ob.cell.fix_all_cell = True
    return c_ob


# Set a bunch of calculator settings for CASTEP
def set_settings(atoms, seed, xc_func, settings, ps_type='NCP'):
    ps_type = ps_type.upper()
    # Attach the calculator
    calc = ase.calculators.castep.Castep(keyword_tolerance=1)
    # include interface settings in .param file
    calc._export_settings = True
    calc._pedantic = True
    # Set working directory
    calc._seed = seed
    calc._label = seed
    calc._directory = seed
    # Set the parameters
    calc = cas_in_prep(calc, xc_func, settings)
    # Set the calculator
    atoms.set_calculator(calc)

    # Set the psudopot xc type
    xc_use = xc_func.upper()
    if xc_func == 'PBE' or xc_func == 'PBE0':
        xc_use = 'PBE'
    elif xc_func == 'B3LYP' or xc_func == 'BLYP':
        xc_use = 'BLYP'

    # Use custom pseudo-potential
    atoms.calc.set_pspot('%s19_%s_OTF' % (ps_type, xc_use))
    return atoms


def cal_attach(atoms, calc, calc_args):
    """
    Attaches calculators to an atoms object, can be used to attach
    :param atoms:
    :param calc:
    :param calc_args:
    :return:
    """
    calc = calc.upper()
    if calc == 'EMT':
        atoms.set_calculator(EMT())
    elif calc == 'CASTEP':
        # print('Using Castep')
        atoms = set_settings(atoms, *calc_args)
    elif calc == 'ONETEP':
        # print('Using Onetep')
        calc = onetep_settings(*calc_args)
        atoms.set_calculator(calc)
    else:
        exit('Problem with calculator name!')
    return atoms


def onetep_imp_sol(calc):
    """
    ONETEP implicit solvent settings
    Taken from:
    https://www.onetep.org/pmwiki/uploads/Main/MasterClass2019/impsolventCC-GAB.pdf
    :param calc:
    :return:
    """
    calc.set(is_implicit_solvent=True)
    calc.set(is_smeared_ion_rep=True)
    calc.set(is_include_apolar=True)
    calc.set(mg_defco_fd_order=8)
    calc.set(is_autosolvation=True)
    calc.set(is_dielectric_model="FIX_INITIAL")  # SELF_CONSISTENT

    # calc.set(is_solvent_surf_tension="0.0000133859 ha/bohr**2")
    # calc.set(is_density_threshold=0.00035)
    # calc.set(is_solvation_beta=1.3)
    # calc.set()
    return calc


def onetep_settings(seed='data', xc_func='pbe', ps_type='abinit', f_imp_sol=False, e_cut=600, f_paw=True, p_t=None,
                    disper=0):
    """

    :param seed:
    :param xc_func:
    :param ps_type:
    :param p_t:
    :param f_imp_sol:
    :param e_cut:
    :return:
    """
    # Set up a command to invoke the ONETEP calculator
    if p_t is None:  # on 4 processes of 6 threads each
        procs = 20
        threads = 1
    else:  # Pick up from command line
        procs = p_t[0]
        threads = p_t[1]

    ps_pot_path = r'/mnt/beegfs/users/ls00338/ps_pots/'
    """
    onetep_cmd = '/users/ls00338/onetep/bin/onetep.eureka'
    environ[
        "ASE_ONETEP_COMMAND"] = f'export OMP_NUM_THREADS={threads}; mpirun -n {procs} {onetep_cmd} PREFIX.dat >> PREFIX.out 2> PREFIX.err'
    """
    # Determine the calculator
    calc = Onetep(label=seed)

    # Sort the ps pots
    if ps_type == 'abinit':
        calc.set(pseudo_path=os.path.join(ps_pot_path, 'JTH-PBE-atomicdata-1.0'))
        calc.set(pseudo_suffix=r'.PBE-paw.abinit')
    elif ps_type == 'recpot':
        calc.set(pseudo_path=ps_pot_path)
        calc.set(pseudo_suffix=r'-onetep.recpot')
        f_paw = False
    else:
        exit('ps type not recognised!')

    # Core settings
    calc.set(paw=f_paw, xc=xc_func, cutoff_energy=str(e_cut) + ' eV', dispersion=disper)
    # calc.set(ngwf_threshold_orig=1.0e-6,elec_energy_tol="1.0e-5 eV",elec_force_tol ="1.0e-2 eV/ang")
    if f_imp_sol:
        onetep_imp_sol(calc)
    return calc


def taut_mid(atoms, idx):
    """
    For a given input of atoms object moves the middle index atom to the middle point of the out two idexes
    :param atoms: atoms object
    :param idx: list of indexes [mol1, proton, mol2]
    :return: atoms object with the proton moved
    """
    # Get the atom positions
    moved = atoms.copy()
    pos = moved.get_positions()
    # Find the mid point location
    mid = 0.5 * (pos[idx[0], :] + pos[idx[2], :])
    # Move the atom
    moved.positions[idx[1]] = mid
    return moved


def neb_folder_cleanup(dir=os.getcwd(), sub='data'):
    """
    Finds subfolders with a given substring in a dir and removes the subfolders
    :param dir: directory to look
    :param sub: substring to filter by
    :return:
    """
    f_list = sub_folder_list(dir, sub)
    print('Folders to be removed:\n', f_list)
    # loop over list and remove each folder
    for i in f_list:
        os.rmdir(i)
    assert len(sub_folder_list(dir, sub)) == 0
    return None


def rm_neb_files(dir=os.getcwd(), neb_str='neb0', exclude=None):
    """
    Removes any neb files, prevents conflicts.
    :param dir: input directory
    :param neb_str: substring used to highlight the neb trajectory files
    :param exclude: list of substrings which you dont want to remove
    :return: list of file paths
    """
    onlyfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    files = [i for i in onlyfiles if neb_str in i]
    if exclude != None:  ### FIX ME
        files = [element for element in files if element not in exclude]
    for i in files:
        print('Removing: ', i)
        os.remove(os.path.join(dir, i))


# Check OS then plot
def os_plot_show(os_name='nt'):
    """
    Checks the system OS. This is to prevent plotting to HPC.
    nt is windows
    :param os_name: The name of the operating system
    :return: None
    """
    # Check if the os is windows
    if os.name == os_name:
        plt.show()
    plt.close()
    return None


# plots the last image from a traj file
def plot_traj(file_traj='prod.traj', rad=.8):
    """
    Plots a .traj file and saves as a pdf to the current working dir
    :param file_traj: input file
    :param rad: radii of each atom
    :return: None
    """
    # remove and re-apply file extenstions
    file_traj = os.path.splitext(file_traj)[0] + '.traj'
    atoms = io.read(file_traj, -1)
    fig, axarr = plt.subplots(1, 4, figsize=(15, 5))
    plot_atoms(atoms, axarr[0], radii=rad, rotation=('0x,0y,0z'))
    plot_atoms(atoms, axarr[1], radii=rad, rotation=('90x,45y,0z'))
    plot_atoms(atoms, axarr[2], radii=rad, rotation=('45x,45y,0z'))
    plot_atoms(atoms, axarr[3], radii=rad, rotation=('90x,0y,0z'))
    axarr[0].set_axis_off()
    axarr[1].set_axis_off()
    axarr[2].set_axis_off()
    axarr[3].set_axis_off()
    fig.savefig(os.path.splitext(file_traj)[0] + ".pdf")
    os_plot_show()
    return None


# converts a .traj to a xyz file
def convert_traj_2_xyz(file_traj='react.traj', file_xyz=None, pick=-1):
    """
    Converts a .traj file to a .xyz formatted file
    :param file_traj: trajectory file
    :param file_xyz: xyz file to save to
    :return:
    """
    # remove and re-apply file extenstions
    # file_traj = os.path.splitext(file_traj)[0] + '.traj'
    # file_xyz = os.path.splitext(file_traj)[0] + '.xyz'

    file_traj = file_traj.split('.')[0] + '.traj'
    if file_xyz is None:
        file_xyz = file_traj.split('.')[0] + '.xyz'
    else:
        # remove and re-apply file ext
        file_xyz = os.path.splitext(file_xyz)[0] + '.xyz'

    # Grab the traj from file
    a = io.read(file_traj + '@:')
    a = a[pick]

    symbols = a.get_chemical_symbols()
    N = len(symbols)
    positions = a.get_positions()
    A = [str(N)]
    A.append(' ')
    # loop over the elements
    for i in range(N):
        # put together the symbol and the xyz on one string line
        A.append(str(symbols[i]) + '   ' + str(positions[i]).strip('[]'))
    # Save the file
    np.savetxt(file_xyz, A, format('%s'))
    return a


def n_plot(xlab, ylab, xs=14, ys=14):
    """
    Makes a plot look nice by introducting ticks, labels, and making it tight
    :param xlab:x axis label
    :param ylab: y axis label
    :param xs: x axis text size
    :param ys: y axis text size
    :return: None
    """
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def file_list(mypath=os.getcwd()):
    """
    List only the files in a directory given by mypath
    :param mypath: specified directory, defaults to current directory
    :return: returns a list of files
    """
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    return onlyfiles


# List only the top level folders in a directory
def folder_list(mypath=os.getcwd()):
    """
    List only the top level folders in a directory given by mypath
    NOTE THIS IS THE SAME AS top_dirs_list
    :param mypath: specified directory, defaults to current directory
    :return: returns a list of folders
    """
    onlyfolders = [f for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]
    return onlyfolders


# List only files which contain a substring
def sub_file_list(mypath, sub_str):
    """
    List only files which contain a given substring
    :param mypath: specified directory
    :param sub_str: string to filter by
    :return: list of files which have been filtered
    """
    return [i for i in file_list(mypath) if sub_str in i]


# List only folders which contain a substring
def sub_folder_list(mypath, sub_str):
    """
    List only folders which contain a given substring
    :param mypath: specified directory
    :param sub_str: string to filter by
    :return: list of folders which have been filtered
    """
    return [i for i in folder_list(mypath) if sub_str in i]


# Bring the path back one
def parent_folder(mypath=os.getcwd()):
    """
    Bring the path back by one
    :param mypath: specified directory, defaults to current directory
    :return: parent path
    """
    return os.path.abspath(os.path.join(mypath, os.pardir))


# Backs up a file if it exists
def file_bck(fpath):
    """
    Backs up a file if it exists
    :param fpath: file to check/backup
    :return: None
    """
    if os.path.exists(fpath) == True:
        bck = fpath.split('.')
        assert len(bck) == 2
        dst = bck[0] + '_bck.' + bck[1]
        shutil.copyfile(fpath, dst)
    return None


# Helper to natural_keys
def atoi(text):
    """
    helper function of natural_keys
    :param text: input text
    :return: ??
    """
    return int(text) if text.isdigit() else text


# Human sorts a list of strings
def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    :param text:
    :return: sorted list
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# calculate fmax
def get_fmax(forces):
    return np.sqrt((forces ** 2).sum(axis=1).max())


def get_phonon_image(pick, atms):
    if pick == 'R':  # Reactant
        print('Using reactant image')
        img = 0
    elif pick == 'P':
        print('Using product image')
        img = -1
    elif pick == 'TS':
        energy = [i.get_potential_energy() for i in atms]
        img = np.argmax(energy)
    else:  # Assume pick is the image you want
        img = pick

    print('Image picked = ', img)
    return str(img)


def image_picker(name, file_path, rtn_idx=False):
    """
    Returns the image of a NEB band from specified name / number

    :param name: Name of the image to pick or number
    :param file_path: path to find the images
    :return: atoms object
    """

    if type(name) == str:
        # Reactant
        if name.lower() == "react" or name.lower() == "r":
            idx = 0
        # TS
        elif name.lower() == "ts":
            img = read(file_path, index=':')
            # Get the energies of the bands
            nebfit = ase.utils.forcecurve.fit_images(img)
            e = nebfit[1]
            idx = int(np.where(e == max(e))[0])
            print("TS image number = ", idx)
        # Product
        elif name.lower() == "prod" or name.lower() == "p":
            idx = -1
        # Specific image
        else:
            idx = name
    else:
        idx = name

    if rtn_idx:
        return read(file_path, index=idx), idx
    else:
        return read(file_path, index=idx)


def remove_pbc(atoms_in, file_tmp="tmp.xyz", f_rm=True):
    """
    This function strips away bc information.

    TODO
    Check for a list of atoms and decide what to do with them!

    Example usage:
    view(remove_pbc(ase.io.read("G_C_B3LYP_GOLD_IMPSOL_NWC.traj")))

    :param atoms_in: atoms object
    :param file_tmp: temporary file which is stored to and removed
    :param f_rm: logic flag to remove temporary file
    :return: atoms object
    """
    # Write a temporary xyz file
    ase.io.write(file_tmp, atoms_in)
    # Load the xyz file
    lines = np.loadtxt(file_tmp, dtype=str, delimiter="\n")
    # Remove the comment with the extra info
    lines[1] = " "
    # Save the modified xyz file
    np.savetxt(file_tmp, lines, delimiter="\n", fmt="%s")
    # Load the modified xyz file to atoms object
    atoms_out = ase.io.read(file_tmp)
    # Delete the temporary xyz file
    if f_rm:
        os.remove(file_tmp)
    # Return the cleaned atoms
    return atoms_out


def traj_clean(name):
    """
    Cleans up an input traj filename for use. Stripping off the .traj@: for example
    :param name: input trajectory file
    :return: cleaned up name
    """
    # Split up by . then take the first part
    tmp = name.split('.')[0]

    # Add the .traj back on
    rtn = tmp + '.traj'
    return rtn


# Routines
class opti_rout(object):
    def __init__(self, file_name):
        """
        Example usage
        file_path = r'/users/ls00338/xyz/ds_TGT.traj'
        ob = ase_common.opti_rout(file_path)
        # Get the atoms
        atoms = ob.get_atoms()
        # Get the calculator settings
        calc_args = (seed, xc_func, ps_type, f_imp_sol, e_cut, True, None, disper)
        # Attach the calculator
        ase_common.cal_attach(atoms, 'onetep', calc_args)
        print('Relaxing initial structure', flush=f_flush)
        # Minimise!
        ob.line_switch(atoms)


        :param file_name:
        """
        # Init variables
        self.calc_name = None
        self.xc_func = None
        self.settings = None

        self.file_name = file_name
        self.f_flush = True
        self.seed = 'data'
        self.logfile = '-'
        self.fname = 'data'
        self.fname_s = self.fname + '_s'
        self.fmax = 0.01
        self.fmax_s = 1.0
        self.vac = 10
        self.xyz_folder = r'/users/ls00338/xyz/'

    def get_atoms(self):
        # Get the atoms
        atoms = io.read(os.path.join(self.xyz_folder, self.file_name))
        atoms.get_cell()
        atoms.center(vacuum=self.vac)
        return atoms

    def ini_standard_calc(self, atoms):
        self.calc_name = 'castep'
        self.settings = 'gold_impsol'
        self.xc_func = 'pbe'
        self.ps_type = 'NCP'
        calc_args = (self.seed, self.xc_func, self.settings, self.ps_type)
        cal_attach(atoms, self.calc_name, calc_args)
        return atoms

    def relax(self, atm):
        dyn = BFGS(atm, trajectory=self.fname + '.traj', logfile=self.fname + '.log', restart=self.fname + '.pckl')
        dyn.run(fmax=self.fmax)

    def fire(self, atm):
        dyn = FIRE(atm, trajectory=self.fname + '.traj', logfile=self.fname + '.log', restart=self.fname + '.pckl')
        dyn.run(fmax=self.fmax)

    def precon(self, atm):
        # Run preconditioned
        opt = PreconLBFGS(atm, precon=Exp(A=3), use_armijo=True,
                          trajectory=self.fname_s + '.traj',
                          logfile=self.fname_s + '.log',
                          restart=self.fname_s + '.pckl')
        opt.run(fmax=self.fmax_s)

    def gaus_pro(self, atm):
        dyn = GPMin(atm, trajectory=self.fname + '.traj',
                    logfile=self.fname + '.log',
                    restart=self.fname + '.pckl')
        dyn.run(fmax=self.fmax)

    def line(self, atm):
        dyn = BFGSLineSearch(atm, trajectory=self.fname + '.traj', logfile=self.fname + '.log')
        dyn.run(fmax=self.fmax)

    def line_switch(self, atm):
        dyn = BFGS(atm, trajectory=self.fname_s + '.traj', logfile=self.fname_s + '.log')
        dyn.run(fmax=self.fmax_s)
        dyn1 = BFGSLineSearch(atm, trajectory=self.fname + '.traj', logfile=self.fname + '.log')
        dyn1.run(fmax=self.fmax)

    def low_line_switch(self, atm):
        dyn = LBFGS(atm, trajectory=self.fname_s + '.traj', logfile=self.fname_s + '.log')
        dyn.run(fmax=self.fmax_s)
        dyn1 = LBFGSLineSearch(atm, trajectory=self.fname + '.traj', logfile=self.fname + '.log')
        dyn1.run(fmax=self.fmax)

    def fire_switch(self, atm):
        dyn = FIRE(atm, trajectory=self.fname_s + '.traj', logfile=self.fname_s + '.log')
        dyn.run(fmax=self.fmax_s)
        dyn1 = LBFGS(atm, trajectory=self.fname + '.traj', logfile=self.fname + '.log')
        dyn1.run(fmax=self.fmax)

    def precon_switch(self, atm):
        # Run preconditioned
        dyn = PreconLBFGS(atm, precon=Exp(A=3), use_armijo=True,
                          trajectory=self.fname_s + '.traj',
                          logfile=self.fname_s + '.log')
        dyn.run(fmax=self.fmax_s)
        # Switch to normal
        dyn1 = LBFGS(atm, trajectory=self.fname + '.traj',
                     logfile=self.fname + '.log')
        dyn1.run(fmax=self.fmax)

    def gaus_switch(self, atm):
        dyn = GPMin(atm, trajectory=self.fname_s + '.traj',
                    logfile=self.fname_s + '.log',
                    restart=self.fname_s + '.pckl')
        dyn.run(fmax=self.fmax_s)
        # Switch to normal
        dyn = LBFGS(atm, trajectory=self.fname + '.traj',
                    logfile=self.fname + '.log',
                    restart=self.fname + '.pckl')
        dyn.run(fmax=self.fmax)


class dft_calc(object):
    def __init__(self, calc_type, seed, xc_f, settings, ele, ps_type='NCP', charge=None, mult=1, unixsocket=None):
        """
        defines the CASTEP and ONETEP DFT code calculator

        """
        # Default to CASTEP
        if calc_type is None:
            self.calc_type = 'CASTEP'
        # Default to name data
        if seed is None:
            self.seed = 'data'
        if xc_f is None:
            self.xc_f = 'PBE'
        self.unixsocket = unixsocket
        self.f_flush = True

        self.calc = None
        self.f_paw = False
        self.e_cut = 400
        self.basis_set = "3-21G"
        self.xc_f = xc_f.upper()
        self.calc_type = calc_type.upper()
        self.ang_2_bohr = 1.88973

        # Set the charge if it is None, determined by the number of Ps
        self.charge = recommend_charge(ele, charge)
        self.ele_list = ele
        self.mult = mult

        self.settings = settings
        self.seed = seed

        if self.calc_type == 'CASTEP':
            self.ps_type = ps_type
            # self.ps_type = 'C'
            self.ps_ext = '.usp'
            tmp = None
            if self.ps_type.upper() == 'NCP':
                if self.xc_f == 'PBE' or self.xc_f == 'PBE0':
                    tmp = 'PBE'
                elif self.xc_f == 'B3LYP' or self.xc_f == 'BLYP':
                    tmp = 'BLYP'
                elif self.xc_f == 'LDA':
                    tmp = 'LDA'
                else:
                    exit('problem with xc and not supporting ps type...')
                self.ps_suff = '_%s19_%s_OTF%s' % (self.ps_type, tmp, self.ps_ext)
            elif self.ps_type.upper() == 'SP':
                self.ps_suff = '_00%s' % self.ps_ext
            else:
                exit('problem with ps type...')
        elif self.calc_type == 'ONETEP':
            self.calc = None
            self.ps_type = 'recpot'
        elif self.calc_type == 'EMT':
            self.calc = EMT()
        elif self.calc_type == "NWCHEM":
            pass
        elif self.calc_type == "NWCHEMDNA":
            pass
        else:
            exit('Calculator not supported')
        print('Calculator init: ' + self.calc_type, flush=self.f_flush)

    def cas_calc(self):
        assert self.calc_type == 'CASTEP'
        self.calc = Castep(keyword_tolerance=1)

        # include interface settings in .param file
        self.calc._export_settings = True
        self.calc._pedantic = True
        # Set working directory
        self.calc._seed = self.seed
        self.calc._label = self.seed
        self.calc._directory = self.seed
        self.calc._link_pspots = True  # True
        self.calc._copy_pspots = False  # False
        self.calc._build_missing_pspots = False  # False
        self.calc.param.reuse = True
        self.calc.param.xc_functional = self.xc_f
        self.calc.param.cut_off_energy = self.e_cut
        self.calc.param.num_dump_cycles = 0
        self.calc.param.fix_occupancy = True
        self.calc.param.max_scf_cycles = 300
        self.calc.param.mixing_scheme = 'pulay'
        self.calc.param.opt_strategy = "speed"
        self.calc.param.max_cg_steps = 5
        self.calc.param.charge = self.charge

        # Set the cell file keywords
        self.calc.cell.symmetry_generate = True
        self.calc.cell.fix_com = False
        self.calc.cell.fix_all_cell = True
        # Set the ps pots
        self.ps_pot_ele()

    def ps_pot_ele(self):
        """
        Loops over the elements and sets the ps pot
        :return:
        """
        for e in self.ele_list:
            self.calc.cell.species_pot = (e, e + self.ps_suff)
        return None

    def cas_in_prep(self):
        """

        :param self.calc:
        :param xc_func:
        :param set: expected: gold, dev, mbd, impsol
        :return:
        """
        assert self.calc_type == 'CASTEP'
        set_list = self.settings.split('_')
        if 'cheap' in set_list:
            print('Using cheap settings', flush=self.f_flush)
            # Set the param file keywords
            self.calc.param.cut_off_energy = 600
            self.calc.param.elec_energy_tol = 1.0E-5
            self.calc.param.elec_eigenvalue_tol = 1.0E-5

        elif 'gold' in set_list:
            print('Using gold settings', flush=self.f_flush)
            # Set the param file keywords
            self.calc.param.cut_off_energy = 1100
            self.calc.param.elec_energy_tol = 1.0E-8
            self.calc.param.elec_eigenvalue_tol = 1.0E-8
            self.calc.param.fine_grid_scale = 2.0
            self.calc.param.mix_charge_amp = 0.6

        elif 'extreme' in set_list:
            print('Using extreme settings', flush=self.f_flush)
            # http://www.castep.org/CASTEP/FAQSCF
            # Set the param file keywords
            self.calc.param.elec_method = 'EDFT'
            self.calc.param.mixing_scheme = 'Broyden'
            self.calc.param.cut_off_energy = 1500
            self.calc.param.elec_energy_tol = 5.0E-7
            self.calc.param.elec_eigenvalue_tol = 5.0E-7
            self.calc.param.grid_scale = 3.0
            self.calc.param.fine_grid_scale = 4.0
            self.calc.param.mix_charge_amp = 0.2
            self.calc.param.mix_cut_off_energy = 1500 * 2

        elif 'sp' in set_list:
            # Set the param file keywords
            # self.calc.param.xc_functional = 'pbe'
            self.calc.param.finite_basis_corr = 0
            self.calc.param.fix_occupancy = False
            self.calc.param.elec_method = 'dm'
            self.calc.param.mixing_scheme = 'pulay'
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'TS'
            self.calc.param.elec_energy_tol = 1.0E-7
            self.calc.param.max_scf_cycles = 300
            self.calc.param.cut_off_energy = 400
            self.calc.param.num_dump_cycles = 0
            self.calc.param.DIPOLE_CORRECTION = "SELF CONSISTENT"
            self.calc.param.SPIN_POLARISED = False
            self.calc.param.write_checkpoint = 'minimal'

            # Cell keywords
            self.calc.param.opt_strategy = "speed"
            self.calc.param.reuse = True
            self.calc.cell.symmetry_generate = True
            self.calc.cell.fix_com = False
            self.calc.cell.fix_all_cell = True
            self.calc.cell.kpoint_mp_grid = [6, 6, 1]

        elif 'defect' in set_list:
            # Set the param file keywords
            self.calc.param.fix_occupancy = False
            self.calc.param.elec_method = 'dm'
            self.calc.param.mixing_scheme = 'pulay'
            # self.calc.param.sedc_apply = True
            # self.calc.param.sedc_scheme = 'TS'

            self.calc.param.elec_energy_tol = 5.0E-6
            self.calc.param.max_scf_cycles = 300
            self.calc.param.cut_off_energy = 300
            self.calc.param.num_dump_cycles = 0
            # self.calc.param.DIPOLE_CORRECTION = "SELF CONSISTENT"
            self.calc.param.SPIN_POLARISED = False
            self.calc.param.write_checkpoint = 'minimal'

            # Cell keywords
            self.calc.param.opt_strategy = "speed"
            self.calc.param.reuse = True

            self.calc.cell.symmetry_generate = True
            self.calc.cell.snap_to_symmetry = True
            self.calc.cell.symmetry_tol = 1e-9

            self.calc.cell.fix_com = False
            self.calc.cell.fix_all_cell = True
            self.calc.cell.kpoint_mp_grid = [4, 4, 1]



        elif 'nick' in set_list:
            # Set the param file keywords
            self.calc.param.xc_functional = 'pbe'
            self.calc.param.grid_scale = 2.0
            self.calc.param.fine_grid_scale = 2.0
            self.calc.param.elec_method = "EDFT"
            self.calc.param.mixing_scheme = "PULAY"
            self.calc.param.fix_occupancy = False
            self.calc.param.cut_off_energy = 400.0
            self.calc.param.opt_strategy = "speed"
            self.calc.param.num_dump_cycles = 0
            self.calc.param.write_formatted_density = False
            self.calc.param.calculate_stress = True
            self.calc.param.geom_force_tol = 0.025
            self.calc.param.geom_stress_tol = 0.1
            self.calc.param.geom_disp_tol = 0.001
            self.calc.param.MAX_SCF_CYCLES = 250
            self.calc.param.SPIN_FIX = 15
            self.calc.param.SPIN_POLARIZED = True
            self.calc.param.GEOM_MAX_ITER = 250
            self.calc.param.SEDC_APPLY = True
            self.calc.param.SEDC_SCHEME = "TS"
            self.calc.param.SPIN = 0.75

            # Extra
            self.calc.param.reuse = True

            # Cell keywords
            self.calc.cell.symmetry_generate = False
            self.calc.cell.fix_com = False
            self.calc.cell.kpoint_mp_grid = [4, 4, 1]

        elif 'rob' in set_list:
            # Set the param file keywords
            self.calc.param.xc_functional = 'pbe'
            self.calc.param.grid_scale = 2.0
            self.calc.param.fine_grid_scale = 2.0
            self.calc.param.elec_method = "EDFT"
            self.calc.param.cut_off_energy = 600.0
            self.calc.param.opt_strategy = "speed"
            self.calc.param.num_dump_cycles = 0
            self.calc.param.MAX_SCF_CYCLES = 250
            self.calc.param.SEDC_APPLY = True
            self.calc.param.SEDC_SCHEME = "G06"
            self.calc.param.write_checkpoint = 'minimal'
            self.calc.param.elec_energy_tol = 1e-5

            # Extra
            self.calc.param.reuse = True

            self.calc.param.fix_occupancy = False

            # Cell keywords
            self.calc.cell.symmetry_generate = False
            self.calc.cell.fix_com = False
            self.calc.cell.kpoint_mp_grid = [6, 6, 1]

        else:
            print('Warning not in settings...', flush=self.f_flush)
            self.calc.param.cut_off_energy = float(self.settings)

        # Check to see if settings are given
        if 'dev' in set_list:
            self.calc.param.devel_code = "improve_wvfn"

        # Dispersion schemes
        # OBS = Phys. Rev. B 73, 205101, (2006)
        # G06 = J. Comput. Chem. 27, 1787, (2006)
        # JCHS = J. Comput. Chem. 28, 555, (2007)
        # TS = Phys. Rev. Lett.,102, 073005 (2009)
        # TSsurf = Phys. Rev. Lett., 108, 146103 (2012)
        # MBD and TSSCS = Phys. Rev. Lett., 108, 236402 (2012)
        # MBD* = J. Chem. Phys. 140, 18A508 (2014)
        # NB Not all XC_FUNCTIONAL values are supported for all schemes - if in doubt use PBE.

        if 'obs' in set_list:
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'OBS'
        if 'g06' in set_list:
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'G06'
        if 'jchs' in set_list:
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'JCHS'
        if 'ts' in set_list:
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'TS'
        if 'tssurf' in set_list:
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'TSsurf'
        if 'mbd' in set_list:
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'MBD'
        if 'mbd*' in set_list:
            self.calc.param.sedc_apply = True
            self.calc.param.sedc_scheme = 'MBD*'

        if 'impsol' in set_list:
            self.calc.param.implicit_solvent_apolar_term = True

    def one_calc(self):
        assert self.calc_type == 'ONETEP'
        # Get me
        ##ps_pot_path = r'/mnt/beegfs/users/ls00338/ps_pots/'
        ps_pot_path = os.environ.get('ONETEP_PP_PATH')
        # Determine the calculator
        self.calc = Onetep(label=self.seed, charge=self.charge)
        # Sort the ps pots
        if self.ps_type == 'abinit':
            self.calc.set(pseudo_path=os.path.join(ps_pot_path, 'JTH-PBE-atomicdata-1.0'))
            self.calc.set(pseudo_suffix=r'.PBE-paw.abinit')
            self.f_paw = True
        elif self.ps_type == 'recpot':
            self.calc.set(pseudo_path=ps_pot_path)
            self.calc.set(pseudo_suffix=r'-onetep.recpot')
            self.f_paw = False
        else:
            exit('ps type not recognised!')

        # Core settings
        self.calc.set(paw=self.f_paw,
                      xc=self.xc_f,
                      write_forces=True)
        # read_tightbox_ngwfs=True,
        # read_denskern=True)

    def one_in_prep(self):
        assert self.calc_type == 'ONETEP'
        set_list = self.settings.split('_')
        if 'cheap' in set_list:
            print('Using cheap settings', flush=self.f_flush)
            ngwf_rad = 8.0
            r_cut = 13.5  # A
            r_cut = np.round(r_cut * self.ang_2_bohr, 3)

            ngwf_rad = np.round(ngwf_rad * self.ang_2_bohr, 3)
            r_cut = np.round(r_cut * self.ang_2_bohr, 3)

            if self.ps_type == 'abinit':
                self.e_cut = 600
            elif self.ps_type == 'recpot':
                self.e_cut = 400
            self.calc.set(cutoff_energy=str(self.e_cut) + ' eV',
                          ngwf_radius=ngwf_rad,
                          kernel_cutoff=str(r_cut) + " bohr",
                          maxit_ngwf_cg=100)

        elif 'gold' in set_list:
            # https://www.onetep.org/Main/Tutorial3
            print('Using gold settings', flush=self.f_flush)
            ngwf_rad = 8.0
            r_cut = 13.5  # A
            r_cut = np.round(r_cut * self.ang_2_bohr, 3)

            if self.ps_type == 'abinit':
                self.e_cut = 1200
            elif self.ps_type == 'recpot':
                self.e_cut = 800
            self.calc.set(cutoff_energy=str(self.e_cut) + ' eV',
                          # ngwf_threshold_orig=1.0e-6,
                          # elec_energy_tol="1.0e-6 eV",
                          ngwf_radius=ngwf_rad,
                          kernel_cutoff=str(r_cut) + " bohr",
                          maxit_ngwf_cg=100)
            # edft=True)

        elif 'extreme' in set_list:
            print('Using extreme settings', flush=self.f_flush)
            ngwf_rad = 3.5  # A
            r_cut = 13.5  # A

            ngwf_rad = np.round(ngwf_rad * self.ang_2_bohr, 3)
            r_cut = np.round(r_cut * self.ang_2_bohr, 3)

            if self.ps_type == 'abinit':
                self.e_cut = 1500
            elif self.ps_type == 'recpot':
                self.e_cut = 1000
            self.calc.set(cutoff_energy=str(self.e_cut) + ' eV',
                          ngwf_threshold_orig=1.0e-7,
                          elec_energy_tol="1.0e-8 eV",
                          ngwf_radius=ngwf_rad,
                          kernel_cutoff=str(r_cut) + " bohr",
                          maxit_ngwf_cg=500)

        else:
            print('Warning not in settings...', flush=self.f_flush)
            self.calc.set(cutoff_energy=str(self.e_cut) + ' eV')

        # Handle dispersion corrections

        if 'disp1' in set_list:
            self.calc.set(dispersion=1)  # From Elstner [J. Chem. Phys. 114(12), 5149-5155]

        elif 'disp2' in set_list:
            self.calc.set(dispersion=2)  # First from Wu and Yang (I) [J. Chem. Phys. 116(2), 515-524, 2002]

        elif 'disp3' in set_list:
            self.calc.set(dispersion=3)  # Second from Wu and Yang (II) [J. Chem. Phys. 116(2), 515-524, 2002]

        elif 'disp4' in set_list:
            self.calc.set(dispersion=4)  # D2 Grimme [ S. Grimme, J. Comput. Chem. 27(15), 1787-1799, 2006]

        if 'impsol' in set_list:
            self.calc.set(is_implicit_solvent=True)
            self.calc.set(is_smeared_ion_rep=True)
            self.calc.set(is_include_apolar=True)
            # self.calc.set(mg_defco_fd_order=8)
            self.calc.set(is_autosolvation=True)
            self.calc.set(is_dielectric_model="FIX_INITIAL")  # SELF_CONSISTENT
            # self.calc.set(is_solvent_surf_tension="0.0000133859 ha/bohr**2")
            # self.calc.set(is_density_threshold=0.00035)
            # self.calc.set(is_solvation_beta=1.3)

        # HFx required
        if self.xc_f in ["B1LYP", "B1PW91", "B3LYP", "B3PW91", "PBE0", "X3LYP"]:
            # For help see https://www.onetep.org/pmwiki/uploads/Main/Documentation/hfx_2020_v2-00.pdf
            self.calc.set(swri="for_hfx 3 10 V 10 10 WE")  # possibly change last two numbers to 12 if problem
            self.calc.set(hfx_use_ri="for_hfx")
            self.calc.set(hfx_max_l=3)
            self.calc.set(hfx_max_q=10)

    # TOP LEVEL
    def cas_full(self):
        assert self.calc_type == 'CASTEP'
        self.cas_calc()
        self.cas_in_prep()
        return self.calc

    def one_full(self):
        assert self.calc_type == 'ONETEP'
        self.one_calc()
        self.one_in_prep()
        return self.calc

    def nwchem_full(self):
        """
        Note here are the default settings for the SCF convergence
        https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html
         CONVERGENCE
             [energy <real energy default 1e-6>] \
             [density <real density default 1e-5>] \
             [gradient <real gradient default 5e-4>] \
             [hl_tol <real hl_tol default 0.1>]
             [dampon <real dampon default 0.0>] \
             [dampoff <real dampoff default 0.0>] \
             [ncydp <integer ncydp default 2>] \
             [ncyds <integer ncyds default 30>] \
             [ncysh <integer ncysh default 30>] \
             [damp <integer ndamp default 0>] [nodamping] \
             [diison <real diison default 0.0>] \
             [diisoff <real diisoff default 0.0>] \
             [(diis [nfock <integer nfock default 10>]) || nodiis] \
             [levlon <real levlon default 0.0>] \
             [levloff <real levloff default 0.0>] \
             [(lshift <real lshift default 0.5>) || nolevelshifting] \
             [rabuck [n_rabuck <integer n_rabuck default 25>]]

        GRID [xcoarse||coarse||medium||fine||xfine||huge]
        Keyword	Total Energy Target Accuracy
        xcoarse	1e-4
        coarse	1e-5
        medium	1e-6
        fine	1e-7
        xfine	1e-8
        huge	1e-10

        :return: calculator object
        """
        assert self.calc_type == 'NWCHEM'
        set_list = self.settings.split('_')
        if self.unixsocket is None:
            tmp = dict(label='calc/nwchem', charge=self.charge, print="medium")
        else:
            tmp = dict(label='calc/nwchem', charge=self.charge, print="medium", task='optimize',
                       driver={'socket': {'unix': self.unixsocket}})  # 'TIGHT': '',

        do_cosmo_smd = "true"  # Use smd version of cosmo
        # Check if MP2 calculation required
        if self.xc_f == "MP2":
            # https://nwchemgit.github.io/MP2.html
            # https://www.nwchem-sw.org/index-php/Special_AWCforum/st/id219/tolerance_in_CPHF_module.html
            # https://nwchemgit.github.io/Hartree-Fock-Theory-for-Molecules.html#orbital-localization
            tmp = dict(label='calc/nwchem', charge=self.charge, print="medium")  # , task="gradient numerical"
            tmp["mp2"] = dict(freeze=1, tight="")
            # tmp["mp2"] = dict()
            tmp["SCF"] = dict(maxiter=2000)  # , singlet=""
            # tmp["set"] = {'cphf:thresh': '1.0d-3', 'cphf:acc': '1.0d-4'}
            tmp["set"] = {'cphf:thresh': '1.0d-7', 'cphf:acc': '1.0d-7', 'cphf:precond_tol': '1.0d-7'}

            do_cosmo_smd = "false"  # Prevent smd version from being set
        elif self.xc_f == "CAM-B3LYP":
            xc_f_cam_b3lyp = "xcamb88 1.00 lyp 0.81 vwn_5 0.19 hfexch 1.00"
            xc_cam = "0.33 cam_alpha 0.19 cam_beta 0.46"
            # tmp["set int:acc_std"] = 1e-30  # https://sites.google.com/a/ncsu.edu/cjobrien-nwchem-tips/set-directives
            tmp["dft"] = dict(maxiter=2000,
                              iterations=1000,
                              grid="fine nodisk",
                              direct=" ",
                              noio=" ",
                              xc=xc_f_cam_b3lyp,
                              cam=xc_cam,
                              convergence=dict(  # fast="",  # Enable quick guess
                                  energy=1e-9,
                                  density=1e-7,
                                  gradient=1e-6),
                              tolerances=dict(tight="",
                                              accCoul=16,
                                              accqrad=20,
                                              tol_rho=1e-16),
                              mult=self.mult,
                              )
        else:
            # tmp["set int:acc_std"] = 1e-30  # https://sites.google.com/a/ncsu.edu/cjobrien-nwchem-tips/set-directives
            tmp["dft"] = dict(maxiter=2000,
                              iterations=1000,
                              grid="fine nodisk",
                              direct=" ",
                              noio=" ",
                              xc=str(self.xc_f).upper(),
                              convergence=dict(  # fast="",  # Enable quick guess
                                  energy=1e-9,
                                  density=1e-7,
                                  gradient=1e-6),
                              tolerances=dict(tight="",
                                              accCoul=16,
                                              accqrad=20,
                                              tol_rho=1e-16),
                              mult=self.mult,
                              )

        # Determine basis https://nwchemgit.github.io/AvailableBasisSets.html
        if 'cheap' in set_list:
            tmp["basis"] = "3-21G"
        elif 'gold' in set_list:
            tmp["basis"] = "6-311++G**"
        elif 'extreme' in set_list:
            tmp["basis"] = "aug-cc-pVTZ"
        elif 'extreme1' in set_list:
            tmp["basis"] = "cc-pVQZ"
        else:
            tmp["basis"] = str(self.basis_set).upper()

        # Determine dispersion
        if 'disp' in set_list:
            # https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html#disp-empirical-long-range-contribution-vdw
            val = tmp.get("dft")  # Key the key value
            val["disp"] = "vdw 3"  # modify the value
            tmp["dft"] = val  # Put it back

        # Determine dispersion
        if 'disp-xdm' in set_list:
            val = tmp.get("dft")  # Key the key value
            val["xdm "] = "a1 0.6224 a2 1.7068"  # modify the value
            tmp["dft"] = val  # Put it back
            tmp["set"] = {'dft:xcreplicated': False, 'dft:xdmsave': False, 'dft:converged': False}

        # Determine implicit solvent
        if 'n-smd' in set_list:
            do_cosmo_smd = "false"  # Prevent smd version from being set

        if 'impsol' in set_list:
            tmp["cosmo"] = dict(do_cosmo_smd=do_cosmo_smd, solvent="water")

        if 'impsol-low' in set_list:
            tmp["cosmo"] = dict(do_cosmo_smd=do_cosmo_smd, dielec=8.0)

        # Check if free energy calculation is required
        if 'free' in set_list:
            tmp["task"] = "Freq"

        return NWChem(**tmp)

    def nwchemdna_full(self):
        """
        https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html
        :return:
        """
        assert self.calc_type == 'NWCHEMDNA'
        set_list = self.settings.split('_')

        if self.unixsocket is None:
            tmp = dict(label='calc/nwchem', charge=self.charge, print="medium")
        else:
            tmp = dict(label='calc/nwchem', charge=self.charge, print="medium", task='optimize',
                       driver={'socket': {'unix': self.unixsocket}})  # 'TIGHT': '',

        if self.mult is None or self.mult == 1:
            tmp["dft"] = dict(convergence="diis 10  ncydp 0 damp 45 dampon 1d99 dampoff 1d-4",
                              maxiter=2000,
                              iterations=1000,
                              xc=str(self.xc_f).upper(),
                              xdm="a1 0.6224 a2 1.7068",
                              tolerances="accAOfunc 25   accCoul 15 ",
                              grid="fine nodisk",
                              noio=" ",
                              direct=" ",
                              )
        else:
            tmp["dft"] = dict(convergence="diis 10  ncydp 0 damp 45 dampon 1d99 dampoff 1d-4",
                              maxiter=2000,
                              iterations=1000,
                              xc=str(self.xc_f).upper(),
                              xdm="a1 0.6224 a2 1.7068",
                              tolerances="accAOfunc 25   accCoul 15 ",
                              grid="fine nodisk",
                              noio=" ",
                              direct=" ",
                              mult=self.mult,
                              )
        # Determine basis https://nwchemgit.github.io/AvailableBasisSets.html
        if 'cheap' in set_list:
            tmp["basis"] = "3-21G"
        elif 'gold' in set_list:
            tmp["basis"] = "6-311++G**"
        elif 'gold1' in set_list:
            tmp["basis"] = "6-31++G**"
        elif 'extreme' in set_list:
            tmp["basis"] = "aug-cc-pVTZ"
        else:
            tmp["basis"] = "3-21G"

        # Determine implicit solvent
        if 'impsol' in set_list:
            tmp["cosmo"] = dict(do_cosmo_smd="true", solvent="water")

        # Determine implicit solvent
        if 'impsol-low' in set_list:
            tmp["cosmo"] = dict(do_cosmo_smd="true", dielec=8.0)

        tmp["set"] = {'dft:xcreplicated': False, 'dft:xdmsave': False, 'dft:converged': False,
                      'quickguess': False}  # True
        # Quickquess example https://nwchemgit.github.io/Special_AWCforum/st/id3617/task_gradient_failed.html
        # https://nwchemgit.github.io/Density-Functional-Theory-for-Molecules.html#convergence-scf-convergence-control
        calc = NWChem(**tmp)
        return calc

    def give_me(self):
        rtn = None
        if self.calc_type == 'CASTEP':
            rtn = self.cas_full()
        elif self.calc_type == 'ONETEP':
            rtn = self.one_full()
        elif self.calc_type == 'NWCHEM':
            rtn = self.nwchem_full()
        elif self.calc_type == 'NWCHEMDNA':
            rtn = self.nwchemdna_full()
        else:
            exit('Calculator not supported')
        return rtn


class path_finder(object):
    def __init__(self, atoms, dis_index, dis_vector, traj_files=None):
        """
        Class which combines TS eigen-following and minima locating.
        See:
        https://wiki.fysik.dtu.dk/ase/ase/dimer.html


        TODO
        > Mask (to freeze atoms) could be an input
        > Need a check to make sure the initial and finial connected minima are not the same, if they are possibly consider
        starting in a new configuration.

        > Need to play with the internal settings to optimise.

        > Options to feed additional problem specific settings, such as the H bond lengths to use as the maximum dimer
        seperation

        > Try with extrapolate_forces = True to reduce the force calls

        :param atoms: class which contains all the atoms of the initial config, assumed that the calculator is attached
        :param dis_index: Index of the atoms to displace
        :param dis_vector: List of vectors (len(dis_index) x 3)
        :param saddle: Input a previously found saddle point
        :param dimer_direction: Input a previously found saddle direction vector
        :param log_file: Log files, '-' is a good option
        :param traj_files: Trajectory files, must be None or a list
        :param fdstep: Amount to step away from the TS to minimise from. Slow to minimise if small. Can fail otherwise
        :param fmax: Max force norm to minimise to
        :param f_verbosity: Flag to check the verbosity of output
        :param f_flush: Bool, Flag to check if to flush print statements. Overrides f_verbosity if true!
        """

        # atoms stuff and calculators
        if traj_files is None:
            traj_files = ['m1.traj', 's1.traj', 'm2.traj']
        self.log_file = '-'
        self.traj_files = traj_files
        self.atoms = atoms

        # Dimer displacement stuff
        self.dis_index = dis_index
        self.dis_vector = dis_vector

        # Parameters
        self.fdstep = 0.2
        self.fmax = 0.01
        self.max_trans = 0.1
        self.dim_sep = 0.1
        self.f_verbosity = 1
        self.f_flush = False
        self.f_norm = False

        # Dummy
        self.saddle = None
        self.dimer_direction = None

        # Assertions
        assert type(traj_files) == list or traj_files is None, 'traj file names must be None or a list'

    def normalise(self, a):
        """
        Normalise input vector by dividing by the matrix norm
        :param a: Input vector
        :return:
        """
        a = np.array(a)
        return a / np.linalg.norm(a)

    def find_saddle(self):
        """

        :return:
        """
        # Set up the dimer method
        if self.f_verbosity >= 1:
            print('Constructing dimer method', flush=self.f_flush)
        # Make a mask to determine which atoms to move, in this case all can move
        dimer_mask = np.ones(len(self.atoms))
        dimer_mask = [1] * len(self.atoms)
        if self.f_verbosity >= 1:
            print('Dimer mask:', dimer_mask, flush=self.f_flush)
        d_control = DimerControl(initial_eigenmode_method='displacement',
                                 displacement_method='vector',
                                 logfile='-',
                                 mask=dimer_mask,
                                 maximum_translation=self.max_trans,
                                 dimer_separation=self.dim_sep,
                                 extrapolate_forces=True)
        # Configure dimer atoms
        if self.f_verbosity >= 1:
            print('Configuring dimer atoms', flush=self.f_flush)
        d_atoms = MinModeAtoms(self.atoms, d_control, mask=dimer_mask)
        # Construct a zero 3xnum atom array
        displacement_vector = [[0.0] * 3] * (len(self.atoms))
        # Check if it is a list
        if type(self.dis_index) == list:
            assert len(self.dis_index) == len(self.dis_vector)
            # Loop over the atom indexes to displace
            for i in range(len(self.dis_index)):
                displacement_vector[self.dis_index[i]] = self.dis_vector[i]
        else:
            # Just displace the one
            displacement_vector[self.dis_index] = self.dis_vector
        # Set up the displacement
        displacement_vector = self.fdstep * self.normalise(displacement_vector)

        # set dimer direction, optional but helps re-converge
        d_atoms.displace(displacement_vector=displacement_vector)
        d_atoms.initialize_eigenmodes(method='displacement')  # set by displacements
        # Setting the eigenmode explicitly, must be normalized for good results,
        # negative sign will find the saddle backwards
        d_atoms.set_eigenmode(self.normalise(displacement_vector))

        # Converge to a saddle point
        if self.f_verbosity >= 1:
            print('Converge to a saddle point', flush=self.f_flush)
        dim_rlx = MinModeTranslate(d_atoms, trajectory=self.traj_files[1], logfile=self.log_file)
        dim_rlx.run(fmax=self.fmax)
        # Get the atoms
        saddle = d_atoms.get_atoms()
        # Get dimer direction or eigenmode, returns normalized
        dimer_direction = d_atoms.get_eigenmode()
        # Confirm saddle
        if self.f_verbosity >= 1:
            print('force at saddle:', np.linalg.norm(saddle.get_forces()), flush=self.f_flush)
        return saddle, dimer_direction

    def connect(self):
        """

        :param saddle:
        :param dimer_direction:
        :return:
        """
        # relax forward through the saddle
        if self.f_verbosity >= 1:
            print('Relax forward through the saddle', flush=self.f_flush)
        displacement_vector = self.fdstep * self.normalise(self.dimer_direction)
        # Grab the atoms
        con_atoms = self.saddle.copy()
        # Displace the atoms
        con_atoms.set_positions(con_atoms.get_positions() + displacement_vector)

        # Minimise
        BFGS(con_atoms, trajectory=self.traj_files[0], logfile=self.log_file).run(fmax=self.fmax)
        minimum1 = con_atoms.copy()

        # relax backwards through the saddle
        if self.f_verbosity >= 1:
            print('Relax backwards through the saddle', flush=self.f_flush)
        displacement_vector = -self.fdstep * self.normalise(self.dimer_direction)
        # Grab the atoms
        con_atoms = self.saddle.copy()
        # Displace the atoms
        con_atoms.set_positions(con_atoms.get_positions() + displacement_vector)
        # Minimise
        BFGS(con_atoms, trajectory=self.traj_files[2], logfile=self.log_file).run(fmax=self.fmax)
        minimum2 = con_atoms.copy()
        return minimum1, minimum2

    def saddle_vec(self):
        """
        Finds the vector of the saddle point
        :return:
        """
        vec = self.dimer_direction[self.dis_index]
        return vec

    def run_saddle(self):
        """
        Finds a saddle point
        :return:
        """
        if self.f_verbosity >= 1:
            print("Running saddle find", flush=self.f_flush)
        saddle, dimer_direction = self.find_saddle()
        return saddle, dimer_direction

    def run_connect(self):
        """
        Connects two minima from a given starting saddle
        :return:
        """
        if self.f_verbosity >= 1:
            print("Running connection", flush=self.f_flush)
        minimum1, minimum2 = self.connect()
        return minimum1, minimum2

    def run(self):
        """
        Finds a saddle point then connects two minima from a given starting saddle
        :return: minimum1, saddle, minimum2 and the saddle_vector
        """
        saddle, dimer_direction = self.run_saddle()

        self.saddle = saddle
        self.dimer_direction = dimer_direction

        minimum1, minimum2 = self.run_connect()
        return [minimum1, saddle, minimum2], self.saddle_vec()


def cheap_nwchem(xc_f, ele, f_solvent=False, f_disp=False, mem="1 gb", charge=None):
    # Set the charge if it is None, determined by the number of Ps
    charge = recommend_charge(ele, charge)

    tmp = dict(label='calc/nwchem', memory=mem, charge=charge)
    tmp["dft"] = dict(maxiter=2000,
                      direct=" ",
                      xc=str(xc_f).upper(),
                      print="medium",
                      xdm="a1 0.6224 a2 1.7068",
                      )
    tmp["basis"] = "3-21G"  # "6-311++G**"
    if f_solvent:
        # tmp["cosmo"] = dict(do_cosmo_smd="true", solvent="water") #
        tmp["cosmo"] = dict(do_cosmo_smd="true", dielec=8.0)
    if f_disp:
        val = tmp.get("dft")  # Key the key value
        val["disp"] = "vdw 3"  # modify the value
        tmp["dft"] = val  # Put it back
    calc = NWChem(**tmp)
    return calc


def dna_nwchem(xc_f, ele, f_solvent=True, charge=None):
    # Set the charge if it is None, determined by the number of Ps
    charge = recommend_charge(ele, charge)

    tmp = dict(label='calc/nwchem', charge=charge, print="medium")
    tmp["dft"] = dict(convergence="diis 10",
                      maxiter=2000,
                      iterations=100,
                      xc=str(xc_f).upper(),
                      xdm="a1 0.6224 a2 1.7068",
                      tolerances="accAOfunc 25   accCoul 15 ",
                      grid="medium nodisk",
                      noio=" ",
                      )
    # Set the basis set
    # tmp["basis"] = "aug-cc-pvdz"
    tmp["basis"] = "6-311++G**"

    if f_solvent:
        # tmp["cosmo"] = dict(do_cosmo_smd="true", solvent="water") #
        tmp["cosmo"] = dict(do_cosmo_smd="true", dielec=8.0)

    tmp["set"] = {'dft:xcreplicated': False, 'dft:xdmsave': False, 'dft:converged': False}

    calc = NWChem(**tmp)
    return calc


def stitch_images(prototype, xyz_folder, save_file="stitch", f_view=False):
    # Attempt to load all the images and stitch them into one
    # prototype = "opti_in_G_enol_C_imino_sep_1_1_"
    # prototype = "opti_in_G_C_sep_0_0_"
    files = glob.glob(prototype + "?.traj")

    # Load the first one
    atm_path = os.path.join(xyz_folder, files[0])
    atm = read(atm_path, index=-1)
    images = [atm]

    # Loop over all the other files
    for i, file in enumerate(files):
        if i == 0:
            continue
        print(file)
        atm_path = os.path.join(xyz_folder, file)
        atm = read(atm_path, index=-1)
        images += [atm]

    if f_view:
        view(images)

    # Save the files
    save_path = os.path.join(xyz_folder, save_file + ".traj")
    ase.io.write(save_path, images, format="traj")
    save_path = os.path.join(xyz_folder, save_file + ".xyz")
    ase.io.write(save_path, images, format="xyz")
    return images


def recommend_charge(ele, charge):
    # Set the charge if it is None, determined by the number of Ps
    if charge is None:
        charge = 0
        for e in ele:
            if e.upper() == "P":
                charge -= 1
    return charge


def pov_render(atoms, fname="atoms.pov", pov_exe=r"C:\Users\ls00338\POV-Ray\v3.7\bin\pvengine64", f_render=True,
               f_remove=True, x_rot=0, y_rot=0, z_rot=0):
    rot = str(x_rot) + "x," + str(y_rot) + "y," + str(z_rot) + "z"  # found using ag: 'view -> rotate'
    f_split = fname.split(".")[0]
    fname = f_split + ".pov"
    # Common kwargs for eps, png, pov
    generic_projection_settings = {
        'rotation': rot,  # text string with rotation (default='' )
        'radii': 1.0,  # float, or a list with one float per atom .85
        'colors': None,  # List: one (r, g, b) tuple per atom
        'show_unit_cell': 2,  # 0, 1, or 2 to not show, show, and show all of cell
    }
    # Extra kwargs only available for povray (All units in angstrom)
    povray_settings = {
        'display': False,  # Display while rendering
        'pause': False,  # Pause when done rendering (only if display)
        'transparent': True,  # Transparent background
        'canvas_width': None,  # Width of canvas in pixels
        'canvas_height': None,  # Height of canvas in pixels
        'camera_dist': 0.0,  # Distance from camera to front atom 50.0
        # 'image_plane': 1.0,  # Distance from front atom to image plane None
        # 'camera_type': 'perspective',  # perspective, ultra_wide_angle
        'point_lights': [],  # [[loc1, color1], [loc2, color2],...]
        # 'area_light': [(2., 3., 40.),  # location
        #                'White',  # color
        #                .7, .7, 3, 3],  # width, height, Nlamps_x, Nlamps_y
        'background': 'White',  # color
        'textures': None,  # Length of atoms list of texture names
        'celllinewidth': 0.1,  # Radius of the cylinders representing the cell
    }
    # Generate pov ray files
    renderer = write(fname, atoms, **generic_projection_settings, povray_settings=povray_settings)
    # Render things
    if f_render:
        renderer.render(povray_executable=pov_exe)
        if f_remove:
            os.remove(f_split + ".pov")
            os.remove(f_split + ".ini")
    return None


def deuterate(atoms):
    symbols = atoms.get_chemical_symbols()
    mass = atoms.get_masses()
    for i, m in enumerate(mass):
        if symbols[i] == "H":
            mass[i] = 2.01410177812
    atoms.set_masses(mass)
    return atoms
