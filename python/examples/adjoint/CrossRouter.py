import numpy as np
import meep as mp

from meep.adjoint import (OptimizationProblem, FluxLine,
                          xHat, yHat, zHat, origin,
                          parameterized_dielectric, fourier_legendre_basis,
                          plane_wave_basis)

##################################################
##################################################
##################################################
class CrossRouter(OptimizationProblem):

    ##################################################
    ##################################################
    ##################################################
    def add_args(self, parser):

        # add new problem-specific arguments
        parser.add_argument('--wh',       type=float, default=1.5,  help='width of horizontal waveguide')
        parser.add_argument('--wv',       type=float, default=1.5,  help='width of vertical waveguide')
        parser.add_argument('--l_stub',   type=float, default=3.0,  help='waveguide input/output stub length')
        parser.add_argument('--eps',      type=float, default=6.0,  help='waveguide permittivity')
        parser.add_argument('--r_design', type=float, default=2.0,  help='design region radius/half-width')
        parser.add_argument('--nr_max',   type=int,   default=2,    help='maximum index of radial Legendre functions')
        parser.add_argument('--kphi_max', type=int,   default=2,    help='maximum frequency of angular sinusoid')
        parser.add_argument('--kxy_max',  type=int,   default=-1,   help='maximum frequency of linear sinusoids')

        # set problem-specific defaults for existing (general) arguments
        parser.set_defaults(fcen=0.5)
        parser.set_defaults(df=0.2)
        parser.set_defaults(dpml=1.0)

    ##################################################
    ##################################################
    ##################################################
    def init_problem(self, args):

        #----------------------------------------
        # size of computational cell
        #----------------------------------------
        lcen       = 1.0/args.fcen
        dpml       = 0.5*lcen if args.dpml==-1.0 else args.dpml
        sx = sy    = dpml + args.l_stub + args.r_design + args.r_design + args.l_stub + dpml
        cell_size  = mp.Vector3(sx, sy, 0.0)

        #----------------------------------------
        #- design region bounding box
        #----------------------------------------
        design_center = origin
        design_size   = mp.Vector3(2.0*args.r_design, 2.0*args.r_design)
        design_region = mp.Volume(center=design_center, size=design_size)

        #----------------------------------------
        #- objective and source regions
        #----------------------------------------
        gap            =  args.l_stub/6.0                  # gap between source region and flux monitor
        d_flux         =  args.r_design + 0.5*args.l_stub  # distance from origin to NSEW flux monitors
        d_source       =  d_flux + gap                     # distance from origin to source
        d_flx2         =  d_flux + 2.0*gap
        l_flux_NS      =  2.0*args.wv
        l_flux_EW      =  2.0*args.wh
        north          =  FluxLine(0.0, +d_flux, l_flux_NS, mp.Y, 'north')
        south          =  FluxLine(0.0, -d_flux, l_flux_NS, mp.Y, 'south')
        east           =  FluxLine(+d_flux, 0.0, l_flux_EW, mp.X, 'east')
        west1          =  FluxLine(-d_flux, 0.0, l_flux_EW, mp.X, 'west1')
        west2          =  FluxLine(-d_flx2, 0.0, l_flux_EW, mp.X, 'west2')

        objective_regions  = [north, south, east, west1, west2]

        source_center  =  mp.Vector3(-d_source, 0.0)
        source_size    =  mp.Vector3(0.0,l_flux_EW)

        #----------------------------------------
        #- optional extra regions for visualization
        #----------------------------------------
        extra_regions  = [mp.Volume(center=origin, size=cell_size)] if args.full_dfts else []

        #----------------------------------------
        # basis set
        #----------------------------------------
        if args.kxy_max == -1:
            basis = fourier_legendre_basis(radius=args.r_design,
                                           nr_max=args.nr_max,
                                           kphi_max=args.kphi_max)
        else:
            basis = plane_wave_basis(2.0*args.r_design, 2.0*args.r_design,
                                     kx_max=args.kxy_max, ky_max=args.kxy_max)

        #----------------------------------------
        #- objective function
        #----------------------------------------
        fstr=(   'Abs(P1_north)**2 + Abs(M1_south)**2'
               + ' + 0.0*(P1_east + P1_west1 + P1_west2)'
               + ' + 0.0*(M1_north + M1_south + M1_east + M1_west1 + M1_west2)'
               + ' + 0.0*(S_north + S_south + S_east + S_west1 + S_west2)'
             )

        #----------------------------------------
        #- internal storage for variables needed later
        #----------------------------------------
        self.args            = args
        self.dpml            = dpml
        self.cell_size       = cell_size
        self.basis           = basis
        self.design_center   = design_center
        self.design_size     = design_size
        self.source_center   = source_center
        self.source_size     = source_size

        return fstr, objective_regions, extra_regions, design_region, basis

    ##############################################################
    ##############################################################
    ##############################################################
    def create_sim(self, beta_vector, vacuum=False):

        args=self.args

        hwvg=mp.Block(center=origin, material=mp.Medium(epsilon=args.eps),
                      size=mp.Vector3(self.cell_size.x,args.wh))
        vwvg=mp.Block(center=origin, material=mp.Medium(epsilon=args.eps),
                      size=mp.Vector3(args.wv,self.cell_size.y))

        if args.kxy_max == -1:
            router=mp.Cylinder(center=self.design_center, radius=args.r_design,
                               epsilon_func=parameterized_dielectric(self.design_center,
                                                                     self.basis,
                                                                     beta_vector))
        else:
            router=mp.Block(center=self.design_center, size=self.design_size,
                            epsilon_func=parameterized_dielectric(self.design_center,
                                                                  self.basis,
                                                                  beta_vector))
        geometry=[hwvg, vwvg, router]

        envelope = mp.GaussianSource(args.fcen,fwidth=args.df)
        amp=1.0
        if callable(getattr(envelope, "fourier_transform", None)):
            amp /= envelope.fourier_transform(args.fcen)
        sources=[mp.EigenModeSource(src=envelope,
                                    center=self.source_center,
                                    size=self.source_size,
                                    eig_band=args.source_mode,
                                    amplitude=amp
                                   )
                ]

        sim=mp.Simulation(resolution=args.res, cell_size=self.cell_size,
                          boundary_layers=[mp.PML(self.dpml)], geometry=geometry,
                          sources=sources)

        if args.complex_fields:
            sim.force_complex_fields=True

        return sim

######################################################################
# if executed as a script, we look at our own filename to figure out
# the name of the class above, create an instance of this class called
# opt_prob, and call its run() method.
######################################################################
if __name__ == '__main__':
    opt_prob=globals()[__file__.split('/')[-1].split('.')[0]]()
    opt_prob.run()
