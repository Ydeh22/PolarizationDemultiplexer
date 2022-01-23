import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
from mpi4py import MPI
import nlopt
import h5py


"""
TODO
    - fix chunk mismatch problem? (don't worry about this one for now, we have everything we need without saving/loading the structure)
    - setup monitors for 3D sim
    - design binarization (need more than the beta/projection stuff)
    - debug normalization run
"""
#mp.quiet(quietval=True) # deprecated
verbosity = mp.Verbosity()
verbosity.meep = 0      # 0-minimal output, 3-debugging
verbosity.mpb = 0


########################################### Simulation Parameters ###########################################
# make any changes to parameters below, they will be recorded to a 'params.txt' file in the end
######### LOCAL PATH ######### 
#save_dir = '/home/will/optimization_results/nobeta0/'              
######### THORNY PATH #########
save_dir = '/users/wge00001/scratch/optimization_results/nobeta3/' # THORNY
log_name = 'results.h5'
is_3D = False                           # if true, runs 3D simulation, 2D otherwise
plotting = False                        # if true, saves eps plots at each iteration for a 2D sim
resolution = 40
design_resolution = resolution
source_wavelength = 0.93                # 930nm wavelength
fcen = 1/source_wavelength              # c = 1 in meep units
wavelength_bandwidth = 0.01
df = wavelength_bandwidth*(fcen**2)
GaAs = mp.Medium(epsilon = 12.338)      # prevents valid freq range warning

Sx, Sy = 5, 5                           # computational cell size
design_size_x, design_size_y = 1, 1     # 1x1 micron design area (footprint?)
Nx = design_resolution*design_size_x    # design region resolutions:
Ny = design_resolution*design_size_y    
wg_dim = 0.25                           # 250nm waveguides, meep units in microns
buffer_radius = 0.1                     # 'protective' GaAs radius around source
pml_thickness = 1.0                     # boundary layer thickness

min_length = 0.09                       # minimum length scale
eta_i = 0.5                             # intermediate design field thresholding point (0 to 1)
eta_e = 0.55                            # erosion design field thresholding point (0 to 1)
eta_d = 1-eta_e                         # dilation design field thresholding point (0 to 1)
filter_radius = mpa.get_conic_radius_from_eta_e(min_length, eta_e)

cur_beta = 4            # initial value of beta
beta_scale = 1.1        # trying 10% increase to beta instead of doubling
num_betas = 1           
update_factor = 100       # this is the 'new' maxeval (optimizer runs that many iterations for EACH beta, so num_betas*update_factor = # iterations)

random_seed = 11111
#############################################################################################################

def mapping(x, eta, beta):
    filtered_field = mpa.conic_filter(x, filter_radius, design_size_x, design_size_y, design_resolution)
    projected_field = mpa.tanh_projection(filtered_field, beta, eta)
    return projected_field.flatten()

pml_layers = [mp.PML(pml_thickness)]              # rule of thumb: thickness >= 1/2 wavelength
if is_3D:
    # set up 3D geometries/materials
    from meep.materials import SiN, SiO2
    Sz = 5                      
    Nz = 1                      # Nz=1 for fabricable devices
    n = Nx*Ny*Nz                # given to nlopt solver
    GaAs_thickness = 0.19
    SiN_thickness = 0.25
    SiO2_thickness = 3.0
    
    design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny,Nz), GaAs, mp.air, grid_type = 'U_MEAN')
    design_region = mpa.DesignRegion(design_variables, volume = mp.Volume(center = mp.Vector3(0,0,-Sz/2+SiO2_thickness+SiN_thickness+GaAs_thickness/2), size=mp.Vector3(design_size_x, design_size_y, GaAs_thickness)))
    
    wg_x = mp.Vector3(Sx/2-design_size_x/2, wg_dim, GaAs_thickness) # for centering purposes
    wg_y = mp.Vector3(wg_dim, Sy/2-design_size_y/2, GaAs_thickness)

    geometry = [mp.Block(material = mp.air, center = mp.Vector3(0,0,0), size = mp.Vector3(Sx,Sy,Sz)),
                mp.Block(material = SiO2, center = mp.Vector3(Sx,Sy,-Sz/2+SiO2_thickness/2), size = mp.Vector3(Sx,Sy,SiO2_thickness)),                    # base SiO2 layer
                mp.Block(material = SiN,  center = mp.Vector3(Sx,Sy,-Sz/2+SiO2_thickness+SiN_thickness/2), size = mp.Vector3(Sx,Sy,SiN_thickness)),       # SiN layer above that ^^                                  
                mp.Block(material = GaAs, center = design_region.center+mp.Vector3(design_size_x/2+wg_x.x/2,0,0), size = wg_x),                           # waveguide along x
                mp.Block(material = GaAs, center = design_region.center+mp.Vector3(0,design_size_y/2+wg_y.y/2,0), size = wg_y),                           # waveguide along y
                mp.Block(material = design_variables, center = design_region.center, size = design_region.size),                                          # swiss cheese space
                mp.Block(material = design_variables, center = design_region.center, size = design_region.size,
                         e1 = mp.Vector3(x=1).rotate(mp.Vector3(z=1), npa.pi/2), e2 = mp.Vector3(y=1).rotate(mp.Vector3(z=1), -npa.pi/2)),                # rotation for design symmetry
                mp.Cylinder(material = GaAs, center = design_region.center, radius = buffer_radius, height = GaAs_thickness)]                             # protective spacing around QD
else:
    # set up 2D geometries/materials. Same as 3D but only GaAs layer, no thicknesses
    Sz, Nz = 0, 0
    n = Nx*Ny

    design_variables = mp.MaterialGrid(mp.Vector3(Nx,Ny), GaAs, mp.air, grid_type = 'U_MEAN') 
    design_region = mpa.DesignRegion(design_variables, volume = mp.Volume(center = mp.Vector3(0,0,0), size = mp.Vector3(design_size_x,design_size_y,0)))

    wg_x = mp.Vector3(Sx/2-design_size_x/2, wg_dim, 0) 
    wg_y = mp.Vector3(wg_dim, Sy/2-design_size_y/2, 0)

    geometry = [mp.Block(material = mp.air, center = mp.Vector3(0,0,0), size = mp.Vector3(Sx,Sy,Sz)),
                mp.Block(material = GaAs, center = design_region.center+mp.Vector3(design_size_x/2+wg_x.x/2,0,0), size = wg_x),               # waveguide along x
                mp.Block(material = GaAs, center = design_region.center+mp.Vector3(0,design_size_y/2+wg_y.y/2,0), size = wg_y),               # waveguide along y
                mp.Block(material = design_variables, center = design_region.center, size = design_region.size),                              # swiss cheese space
                mp.Block(material = design_variables, center = design_region.center, size = design_region.size,
                         e1 = mp.Vector3(x=1).rotate(mp.Vector3(z=1), npa.pi/2), e2 = mp.Vector3(y=1).rotate(mp.Vector3(z=1), -npa.pi/2)),    # rotation for design symmetry
                mp.Cylinder(material = GaAs, center = design_region.center, radius = buffer_radius, height = 0)]                              # protective spacing around QD

cell = mp.Vector3(Sx, Sy, Sz)

# GaAs QD is essentially TWO orthogonal dipole sources, but we need only ONE due to mirror symmetry
# Using x-polarized source here
gauss_source = [mp.Source(mp.GaussianSource(frequency = fcen, fwidth = df),
                    component = mp.Ex,
                    center = design_region.center,
                    size = mp.Vector3(0,0,0))]

cw_source = [mp.Source(mp.ContinuousSource(wavelength = source_wavelength),
                    component = mp.Ex,
                    center = design_region.center,
                    size = mp.Vector3(0,0,0))]

sim = mp.Simulation(cell_size = cell,
                    geometry = geometry,
                    boundary_layers = pml_layers,
                    sources = gauss_source,
                    resolution = resolution)

# wg monitors centered right before PML boundary (creating them as 'flux regions' which get passed to 'add_mode_monitor' during normalization run)
yp_mon_cen = mp.Vector3(0,(Sy/2+design_size_y)/2-design_size_y/2,0)         # the 'new' one, monitor closer to PML
ym_mon_cen = mp.Vector3(0,-(Sy/2+design_size_y)/2+design_size_y/2,0)
xp_mon_cen = mp.Vector3((Sx/2+design_size_x)/2-design_size_x/2,0,0)
xm_mon_cen = mp.Vector3(-(Sx/2+design_size_x)/2+design_size_x/2,0,0)

#top_wg_fr_cen2 = mp.Vector3(0,(Sy/2+design_size_y/2)/2-design_size_y/2,0)       # the 'old' one, monitor closer to source 
#right_wg_fr_cen2 = mp.Vector3((Sx/2+design_size_x/2)/2-design_size_x/2,0,0)

if is_3D:
    TE_x = mpa.EigenmodeCoefficient(sim, mp.Volume(center = xp_mon_cen, size = mp.Vector3(0, 2*wg_dim, 2*wg_dim)), mode=1)
    TE_y = mpa.EigenmodeCoefficient(sim, mp.Volume(center = yp_mon_cen, size = mp.Vector3(2*wg_dim, 0, 2*wg_dim)), mode=1)
else:
    TE_x = mpa.EigenmodeCoefficient(sim, mp.Volume(center = xp_mon_cen, size = mp.Vector3(0, 2*wg_dim, 0)), mode=1)
    TE_y = mpa.EigenmodeCoefficient(sim, mp.Volume(center = yp_mon_cen, size = mp.Vector3(2*wg_dim, 0, 0)), mode=1)
obj_list = [TE_x, TE_y]


coupling, crosstalk = [], []
counter = [0]
def J(x, y):
    # Objective: for x-pol source maximize power to top waveguide while minimizing power to other waveguide
    if counter[0] == 0:
        coupling.append(np.abs(y)**2)
        crosstalk.append(np.abs(x)**2)
        counter[0] = counter[0]+1
    elif counter[0] == 2:
        counter[0] = 0
    else:
        counter[0] = counter[0]+1

    return (npa.abs(y)**2 - npa.abs(x)**2)      # sticking with examples and using autograd numpy here


opt = mpa.OptimizationProblem(simulation = sim,
                              objective_functions = J,
                              objective_arguments = obj_list,
                              design_regions = [design_region],
                              df = 0,
                              nf = 1,
                              fcen = fcen)

run_num = [0]     # counter to keep track of each iteration
obj_history = []
grad_history = []
eps_history = []
#def f(v, gradient, cur_beta):
def f(v, gradient):
    if mp.am_master(): print("Current iteration: {}".format(run_num[0]+1))
    #print("Current beta: {}".format(cur_beta))
    f0, dJ_du = opt([v])
    #f0, dJ_du = opt([mapping(v, eta_i, cur_beta)])      # compute objective and gradient
    if gradient.size > 0:
        gradient[:] = np.squeeze(dJ_du)
        #gradient[:] = tensor_jacobian_product(mapping,0)(x, eta_i, cur_beta, np.squeeze(dJ_du))    # backprop?
    obj_history.append(np.real(f0))             # record objective function
    grad_history.append(dJ_du)                  # record gradient
    eps_history.append(opt.sim.get_epsilon())   # record permittivty

    # below saves permittivity plots at each iteration to see how design is changing (only for 2D as of now)
    if plotting == True and is_3D == False:
        from matplotlib import pyplot as plt
        plt.ioff()          # turn interactive mode off so plots don't get displayed
        fig = plt.figure()
        ax = plt.gca()
        opt.plot2D(False, ax=ax, plot_sources_flag = False, plot_monitors_flag = False, plot_boundaries_flag = False)
        ax.axis('off')
        plt.savefig(save_dir + 'plots/' + 'iteration_' + str(run_num[0]), dpi=fig.dpi)
    
    run_num[0] = run_num[0]+1
    return np.real(f0)

np.random.seed(random_seed)    # set to ensure all parallel processes get the same initial permittivity distribution
x0 = np.random.rand(n,)         # initial 'guess' of permittivity distribution
#x = npa.ones((n,)) * 0.5

opt.update_design([x0])
algorithm = nlopt.LD_MMA
solver=nlopt.opt(algorithm, n)
solver.set_lower_bounds(0)
solver.set_upper_bounds(1)
solver.set_max_objective(f)
solver.set_maxeval(update_factor)

x = solver.optimize(x0)

'''
for iters in range(num_betas):
    solver = nlopt.opt(algorithm, n)
    solver.set_lower_bounds(0)
    solver.set_upper_bounds(1)
    solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
    solver.set_maxeval(update_factor)
    x[:] = solver.optimize(x)
    cur_beta = cur_beta*beta_scale
'''

def save_params():
    # Saves parameters used in the optimization to a file (only saving the *required* variables for analysis)
    param_dict = {'is_3D':is_3D, 'resolution':resolution, 'design_size_x':design_size_x,
                'design_size_y':design_size_y, 'pml_thickness': pml_thickness,
                'Sx':Sx, 'Sy':Sy, 'Nx':Nx, 'Ny':Ny, 'Sz':Sz, 'Nz':Nz, 'min_length':min_length,
                'num_betas':num_betas, 'update_factor':update_factor, 'random_seed': random_seed,
                'save_dir':save_dir, 'log_name':log_name, 'design_resolution':design_resolution, 'fcen':fcen}
    
    file = open(save_dir+'params.txt', 'w+')
    str_dict = repr(param_dict)
    file.write(str_dict+'\n')
    file.close()
    return

save_params()           # save optimization parameters

# no longer needed:
#opt.sim.dump_structure(fname = save_dir+'eps.h5')           # Save structure(permittivity) to load in future simulations or plotting
#opt.sim.dump_chunk_layout(fname = save_dir+'chunks.h5')     # Save chunk layout too


########################################### Normalization Run ###########################################
nfreq = 100              # Number of frequencies to use when computing flux spectrum
fcen_index = nfreq//2 -1 # center freq index for flux list

if is_3D:
    # flux stuff for 3D geometry
    trash=0 # just to allow code to run (error if just comment left here)
else:
    # Flux region box: (flux computed in + coordinate direction, so subtract xm and ym flux when calculating total flux
    xp_mon = mp.FluxRegion(center = xp_mon_cen, size = mp.Vector3(0,2*xp_mon_cen.x,0))
    yp_mon = mp.FluxRegion(center = yp_mon_cen, size = mp.Vector3(2*yp_mon_cen.y,0,0))
    xm_mon = mp.FluxRegion(center = xm_mon_cen, size = mp.Vector3(0,2*xp_mon_cen.x,0))
    ym_mon = mp.FluxRegion(center = ym_mon_cen, size = mp.Vector3(2*yp_mon_cen.y,0,0))

    xp_flux = sim.add_flux(fcen, df, nfreq, xp_mon)
    yp_flux = sim.add_flux(fcen, df, nfreq, yp_mon)
    xm_flux = sim.add_flux(fcen, df, nfreq, xm_mon)
    ym_flux = sim.add_flux(fcen, df, nfreq, ym_mon)
    
    # checking monitors are in correct place
    '''
    from matplotlib import pyplot as plt
    sim.plot2D(plot_boundaries_flag=True,plot_monitors_flag=True,plot_sources_flag=True,output_plane=mp.Volume(center=mp.Vector3(0,0,0),size=mp.Vector3(Sx,Sy,0)))
    plt.show()
    '''

    sim.use_output_directory(save_dir)      # sends all h5/png output files to savedir

    '''
    animate = mp.Animate2D(sim,
                           fields=mp.Ex,
                           realtime=True,
                           field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none'},
                           boundary_parameters={'hatch':'o','linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3})
    #sim.run(mp.at_every(1, animate), until_after_sources=mp.stop_when_fields_decayed(100, mp.Ex, pt, 1e-3))
    '''
    
    if mp.am_master():
        print('Starting normalization run. . .')

    pt = yp_mon_cen
    sim.restart_fields()
    sim.run(mp.at_beginning(mp.output_epsilon), until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt, 1e-3))
    #opt.sim.run(mp.at_beginning(mp.output_epsilon), mp.to_appended('ex', mp.at_every(2, mp.output_efield_x)), until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt, 1e-3))

    # for quick/local testing
    #sim.run(mp.at_beginning(mp.output_epsilon), until=100)

    # Total power
    Pxp = np.array(mp.get_fluxes(xp_flux))
    Pxm = np.array(mp.get_fluxes(xm_flux))
    Pyp = np.array(mp.get_fluxes(yp_flux))
    Pym = np.array(mp.get_fluxes(ym_flux))
    Ptotal = (Pxp-Pxm)+(Pyp-Pym)

    # Grab eigenmode coefficients to calculate modal powers
    res_x = sim.get_eigenmode_coefficients(xp_flux, [1,2,3,4], eig_parity=mp.NO_PARITY)
    incident_coeffs_x = np.array(res_x.alpha)
    res_y = sim.get_eigenmode_coefficients(yp_flux, [1,2,3,4], eig_parity=mp.NO_PARITY)
    incident_coeffs_y = np.array(res_y.alpha)

    modal_power_x = abs(incident_coeffs_x)**2  #np.abs() takes complex modulus
    modal_power_y = abs(incident_coeffs_y)**2

    wavl = 1/np.array(mp.get_flux_freqs(xp_flux)) * 1000 # convert to nm

    def save_log(opt=None):
        # Saves optimization results for permittivty, gradient, and objectives to single h5 file (w/ data for each iteration)
        hf = h5py.File(save_dir+log_name, 'w', driver = 'mpio', comm = MPI.COMM_WORLD)
        hf.create_dataset('final_eps', data = opt.sim.get_epsilon())          # 'colored' eps data (not the black and white one)
        hf.create_dataset('eps', data = eps_history)
        hf.create_dataset('grad', data = grad_history)
        hf.create_dataset('obj', data = obj_history)
        hf.create_dataset('coupling', data = coupling)
        hf.create_dataset('crosstalk', data = crosstalk)
        hf.create_dataset('mode_power_y', data = modal_power_y)
        hf.create_dataset('mode_power_x', data = modal_power_x)
        hf.create_dataset('ptotal', data = Ptotal)
        hf.create_dataset('wavelength', data = wavl)
        hf.close()
        return

    save_log(opt)
    print('Optimization data saved.')

    '''
    # output results for now to test, eventually will record and use to normalize objective?
    if mp.am_master():
        print()
        print('Top Modal Power: ' + str(top_modal_powers[0,fcen_index,0]))
        print('Right Modal Power: ' + str(right_modal_powers[0,fcen_index,0]))
        print('Total Source "Flux": ' + str(fcen_tot_src_flux))
        print('Top WG Flux: ' + str(fcen_top_flux))
        print('Right WG Flux: ' + str(fcen_right_flux))
        print('-----------------------------------------------')
        print('Total Flux vs Modal Power:')
        print('Top Efficiency: ' + str(top_efficiency1*100))
        print('Right Efficiency: ' + str(right_efficiency1*100))
        print('-----------------------------------------------')
        print('Total Flux vs WG Flux:')
        print('Top Efficiency: ' + str(top_efficiency2*100))
        print('Right Efficiency: ' + str(right_efficiency2*100))
    '''
