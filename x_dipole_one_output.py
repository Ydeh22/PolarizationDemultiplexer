# Simple 2D example optimization to test x-dipole coupling to wavguide along y-direction using parallel meep

import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa, tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from mpi4py import MPI
import h5py

# output control: 0 = minimal, 1 = some, 2 = a lot, 3 = debug
verbosity=mp.Verbosity()
verbosity.meep=0
verbosity.mpb=0

# CHANGE BELOW PARAMETERS MANUALLY #
####################################################################
# I/O parameters #
#save_dir = '/users/wge00001/scratch/optimization_results/test/' #thorny
save_dir = '/home/will/invdes/optimization_results/test/'       #local
log_name = 'results.h5'

# simulation parameters
rand_seed = 12345   # need to specify seed so parallel processes use the same one if using random initial dist.
GaAs = mp.Medium(index=3.45)
resolution = 20
design_region_resolution = 20

waveguide_width = 0.4
waveguide_length = 1.5
design_region_width = 2.0
design_region_height = 2.0
buffer_radius = 0.1
pml_size = 1.0
pml_layers = [mp.PML(pml_size)]

Sx = 2*pml_size+design_region_width+waveguide_length
Sy = 2*pml_size+design_region_height+waveguide_length
cell_size = mp.Vector3(Sx,Sy)

fcen = 1/0.93
wavl_width = 0.01
nfreq = 50
df = fcen**2*wavl_width

freqs = np.array([fcen-df, fcen, fcen+df])
fcen_index = freqs.size//2
src = mp.GaussianSource(frequency=fcen,fwidth=df)
source = [mp.Source(src,
                    component=mp.Ex,
                    center=mp.Vector3(0,0))]

Nx = int(design_region_resolution*design_region_width) + 1  # plus 1 needed to fix 'off by one' error in filter funcs(see git)
Ny = int(design_region_resolution*design_region_height) + 1
design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny),mp.air,GaAs,grid_type='U_MEAN')
design_region = mpa.DesignRegion(design_variables,
                                 volume=mp.Volume(center=mp.Vector3(0,0),
                                                  size=mp.Vector3(design_region_width, design_region_height, 0)))
algorithm = nlopt.LD_MMA
n = Nx * Ny

# initial eps dist:
x = np.ones((n,)) * 0.5

cur_beta = 1
beta_scale=2
num_betas=2
update_factor=5

lb = np.zeros((n,))
ub = np.ones((n,))
####################################################################

# filter parameters and mapping function:
minimum_length = 0.09  # minimum length scale (microns)
eta_i = 0.5            # blueprint (or intermediate) design field thresholding point (between 0 and 1)
eta_e = 0.55           # erosion design field thresholding point (between 0 and 1)
eta_d = 1-eta_e        # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length,eta_e)

def mapping(x,eta,beta):
    # filter
    filtered_field = mpa.conic_filter(x,filter_radius,design_region_width,design_region_height,design_region_resolution)
    
    # projection
    projected_field = mpa.tanh_projection(filtered_field,beta,eta)
    
    # interpolate to actual materials
    return projected_field.flatten()

# output functions (to save parameters & results for later analysis)
def save_params():
    # saves useful parameters used in an optimization to 'params.txt' file
    # ADD MORE HERE (numBeta, pml data, design/cell sizes, etc....)
    param_dict = {'resolution':resolution, 'design_resolution':design_resolution}
    file = open(save_dir+'params.txt', 'w+')
    str_dict = repr(param_dict)
    file.write(str_dict+'\n')
    file.close()
    return

def save_results():
    # saves results of optimization to analyze in separate script/jupyter notebook
    hf = h5py.File(save_dir+log_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    hf.create_dataset('final_eps', data = opt.sim.get_epsilon())
    hf.create_dataset('grad', data = sensitivity)
    hf.create_dataset('obj', data = evaluation_history)
    hf.create_dataset('modal_power', data = modal_power)
    hf.create_dataset('ptotal', data = Ptotal)
    hf.create_dataset('wavelength', data = wavl)
    hf.close()
    return


# with protective cylinder
geometry = [
    mp.Block(center=mp.Vector3(y=Sy/4), material=GaAs, size=mp.Vector3(waveguide_width, Sy/2, 0)),  # vertical waveguide
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables),      # design region
    mp.Block(center=design_region.center, size=design_region.size, material=design_variables,
             e1=mp.Vector3(x=-1), e2=mp.Vector3(y=1)),                                              # enforce mirror symmetry about x=0 plane
    mp.Cylinder(center=design_region.center, material=GaAs, radius=buffer_radius)                   # buffer region to protect QD
]

sim = mp.Simulation(cell_size=cell_size,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=source,
                    resolution=resolution)

# Optimization Problem Setup
mon_center = mp.Vector3(0, design_region_height-pml_size,0)
TE_top = mpa.EigenmodeCoefficient(sim,mp.Volume(center=mon_center,size=mp.Vector3(x=3*waveguide_width)),mode=1)
ob_list = [TE_top]

def J(alpha):
    return npa.abs(alpha)**2

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=J,
    objective_arguments=ob_list,
    design_regions=[design_region],
    frequencies=freqs)


evaluation_history = []
sensitivity = [0]
cur_iter = [0]

def f(v, gradient, cur_beta):
    if mp.am_master(): print("Current iteration: {}".format(cur_iter[0]+1))
    f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])
    if gradient.size > 0:
        gradient[:] = tensor_jacobian_product(mapping,0)(v, eta_i, cur_beta, np.sum(dJ_du, axis=1))
    evaluation_history.append(np.real(f0)[fcen_index])
    sensitivity[0] = dJ_du
    cur_iter[0] = cur_iter[0] + 1
    val = (np.real(f0)).astype(float) # needed to cast to float to prevent nlopt arg. error
    return val[fcen_index]


# Run optimization
for i in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(lb)
    solver.set_upper_bounds(ub)
    solver.set_max_objective(lambda a,g:f(a,g,cur_beta))
    solver.set_maxeval(update_factor)
    x[:] = solver.optimize(x)
    cur_beta = cur_beta*beta_scale

# once optimization is finished, need a normalization run to extract coupling efficiencies:
xp_cen = mp.Vector3(design_region_width-pml_size,0,0)
xm_cen = mp.Vector3(-design_region_width+pml_size,0,0)
yp_cen = mp.Vector3(0,design_region_height-pml_size,0)
ym_cen = mp.Vector3(0,-design_region_height+pml_size,0)

x_size = mp.Vector3(0, design_region_height+pml_size/2,0)
y_size = mp.Vector3(design_region_width+pml_size/2,0,0)

Pxp_mon = mp.FluxRegion(center=xp_cen, size=x_size)                            
xp_flux = sim.add_flux(fcen, df, nfreq, Pxp_mon)
Pxm_mon = mp.FluxRegion(center=xm_cen, size=x_size)                            
xm_flux = sim.add_flux(fcen, df, nfreq, Pxm_mon)
Pyp_mon = mp.FluxRegion(center=yp_cen, size=y_size)                            
yp_flux = sim.add_flux(fcen, df, nfreq, Pyp_mon)
Pym_mon = mp.FluxRegion(center=ym_cen, size=y_size)                            
ym_flux = sim.add_flux(fcen, df, nfreq, Pym_mon)


# Run sim to calculate coupling efficiencies
sim.restart_fields()
sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ex,yp_cen,1e-3))

# Grab total output flux to normalize modal power
Pxp = np.array(mp.get_fluxes(xp_flux))
Pxm = np.array(mp.get_fluxes(xm_flux))
Pyp = np.array(mp.get_fluxes(yp_flux))
Pym = np.array(mp.get_fluxes(ym_flux))
Ptotal = (Pxp-Pxm)+(Pyp-Pym)
#print(Ptotal)

# looking at fundamental mode
res = sim.get_eigenmode_coefficients(yp_flux,[1],eig_parity=mp.NO_PARITY)
incident_coeffs = np.array(res.alpha)

plt.figure()
wavl = 1/np.array(mp.get_flux_freqs(yp_flux))
modal_power = abs(incident_coeffs)**2

save_params()
save_results()