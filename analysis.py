import meep as mp
from matplotlib import pyplot as plt 
import matplotlib as mpl
import numpy as np
import h5py
import ast
'''
 TODO:
    - add in 3D functionality eventually
'''

######### LOCAL (WSL) PATH ######### 
save_dir = '/home/will/optimization_results/nobeta0/'   
            
######### THORNY PATH #########
#save_dir = '/users/wge00001/scratch/optimization_results/' # THORNY

######### LOCAL (WINDOWS) PATH #########  
#save_dir = '/mnt/c/Users/willi/wvu/SQuOL/data/nobeta0/'

log_name = 'results.h5'

# Read in the dictionary from parameter file
file = open(save_dir+'params.txt', 'r')
contents = file.read()
param_dict = ast.literal_eval(contents)
file.close()

# Initialize relevant variables
#save_dir = param_dict['save_dir']
fcen = param_dict['fcen']
is_3D = param_dict['is_3D']
resolution = param_dict['resolution']
design_resolution = param_dict['design_resolution']
Sx, Sy, Sz = param_dict['Sx'], param_dict['Sy'], param_dict['Sz']
Nx, Ny, Nz = param_dict['Nx'], param_dict['Ny'], param_dict['Nz']
num_betas = param_dict['num_betas']
update_factor = param_dict['update_factor']
final_iter = update_factor*num_betas

# Do we need these ones (I don't think so)?
pml_layers = mp.PML(param_dict['pml_thickness'])

'''
# DEBUGGING
print(save_dir)
print(is_3D)
print(resolution)
print(design_resolution)
S_string = str(Sx)+': '+str(Sy)+': '+str(Sz)
N_string = str(Nx)+': '+str(Ny)+': '+str(Nz)
print(S_string)
print(N_string)
print(num_betas)
print(update_factor)
print(pml_layers)
'''


####################### Initialize Sim Object ####################### 
# NOT NEEDED ANYMORE, we have everything we need without dumping/loading structure
'''
sim = mp.Simulation(cell_size = cell,
                    resolution = resolution,
                    boundary_layers = pml_layers,
                    chunk_layout = save_dir+chunk_filename,
                    load_structure = save_dir+structure_filename)

#sim.visualize_chunks()
#sim.plot2D(plot_eps_flag=True, plot_sources_flag=False, plot_monitors_flag=False, plot_boundaries_flag=False)
#plt.show()
'''

####################### Load data from optimization results ####################### 
with h5py.File(save_dir+log_name, 'r') as f:
    # Load data into separate arrays
    objective = list(f['obj'])                  # 1D array, each element is objective value for that iteration
    crosstalk = list(f['crosstalk'])            # 1D array ^^^
    coupling = list(f['coupling'])              # 1D array ^^^
    gradient = list(f['grad'])                  # 3D array? each element is 2D array with the gradient for that iteration
    permittivty = list(f['eps'])                # 3D array? ^^^
    Ptotal = list(f['ptotal'])
    modal_power_x = np.array(f['mode_power_x'])
    modal_power_y = np.array(f['mode_power_y'])
    wavl = np.array(f['wavelength'])

# checking to make sure arrays are not empty.
#print(objective)
#print(crosstalk)
#print(coupling)
#print(gradient)
#print(permittivty)

# Plots the gradient at a specified iteration, defaults to final iteration if not given
def plot_gradient(iteration=final_iter-1):
    grad = gradient[iteration]
    grad_2Darray = np.reshape(grad, (design_resolution, design_resolution))   # convert to 2D array
    grad_2Darray = np.flip(grad_2Darray, axis=0)                              # flip to get correct design orientation
    plt.imshow(grad_2Darray)
    plt.colorbar()
    plt.show()
    return

# Plots the permittivity at a specified iteration, defaults to final iteration if not given:
def plot_permittivity(iteration=final_iter-1):
    eps = permittivty[iteration]
    eps_2Darray = np.reshape(eps, (Sx*resolution, Sy*resolution))         # convert to 2D array
    eps_2Darray = np.flip(eps_2Darray, axis=0)                            # flip to get correct orientation
    plt.imshow(eps_2Darray)
    plt.colorbar()
    plt.show()
    return

# Plots the overall & sub-objectives as a function of iteration
def plot_objective():
    # log scale
    plt.plot(np.log10(objective), 'g', label = '(Log10) Overall Objective')
    plt.plot(np.log10(coupling), 'b--', label = '(Log10) Coupling')
    plt.plot(np.log10(crosstalk), 'r--', label = '(Log10) Crosstalk')
    # regular
    #plt.plot(obj, 'k', label = 'Overall Objective')
    #plt.plot(coupling, 'b', label = 'Coupling')
    #plt.plot(crosstalk, 'r', label = 'Crosstalk')
    plt.legend()
    plt.show()
    return

def plot_x_power():
    plt.figure(1) 
    plt.title('X-WG Modal Power')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('X-WG Modal Power %')
    # last argument in [#,#,#] is prop. direction (0=forward(+), 1=backward(-))
    plt.plot(wavl, modal_power_x[0,:,0]/Ptotal,
             wavl, modal_power_x[1,:,0]/Ptotal,
             wavl, modal_power_x[2,:,0]/Ptotal,
             wavl, modal_power_x[3,:,0]/Ptotal)
    plt.legend(['mode 0', 'mode 1', 'mode 2', 'mode 3'])
    plt.show()

def plot_y_power():
    plt.figure(2)
    plt.title('Y-WG Modal Power')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Y-WG Modal Power %')
    plt.plot(wavl, modal_power_y[0,:,0]/Ptotal,
             wavl, modal_power_y[1,:,0]/Ptotal,
             wavl, modal_power_y[2,:,0]/Ptotal,
             wavl, modal_power_y[3,:,0]/Ptotal)
    plt.legend(['mode 0', 'mode 1', 'mode 2', 'mode 3'])
    plt.show()

# create the plots
plot_gradient()
plot_permittivity()
plot_objective()

plot_x_power()
plot_y_power()

