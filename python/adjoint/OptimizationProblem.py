import sys
import argparse
import numpy as np
import meep as mp

import matplotlib.pyplot as plt

from . import ObjectiveFunction
from . import Visualization
from . import Basis

from .ObjectiveFunction import (ObjectiveFunction, DFTCell, adjoint_options,
                                EHTransverse, Exyz, Hxyz, EHxyz, xHat, yHat,
                                zHat, origin, GridInfo, abs2, rel_diff,
                                get_objective, get_objective_and_gradient)


from .Visualization import visualize_sim, plot_options

######################################################################
# invoke python's 'abstract base class' formalism in a version-agnostic way
######################################################################
from abc import ABCMeta, abstractmethod
ABC = ABCMeta('ABC', (object,), {'__slots__': ()}) # compatible with Python 2 *and* 3:

######################################################################
# OptimizationProblem is an abstract base class from which classes
# describing specific optimization problems should inherit
######################################################################
class OptimizationProblem(ABC):


    ######################################################################
    # pure virtual methods that must be overridden by child
    ######################################################################
    @abstractmethod
    def init_problem(self, args):
        raise NotImplementedError("derived class must implement init_problem() method")


    @abstractmethod
    def create_sim(self, betavector, vacuum=False):
        raise NotImplementedError("derived class must implement create_sim() method")


    ######################################################################
    # virtual methods that *may* optionally be overridden by child
    ######################################################################
    def add_args(self, args):
        pass

    ######################################################################
    ######################################################################
    ######################################################################
    def __init__(self):

        # initialize parser with general arguments
        parser=self.init_args()

        # add problem-specific arguments (if derived class has any)
        self.add_args(parser)

        # parse arguments and do problem-specific initialization first...
        args = self.args = parser.parse_args()
        f_str, objective_regions, extra_regions, design_region, self.basis = self.init_problem(args)

        # ...and now do some general initialization
        fcen,df,nfreq = args.fcen, args.df, args.nfreq
        #self.grid  = FDGrid(size=self.cell_size, res=self.args.res)
        self.objective_cells    = [ DFTCell(region=v,fcen=fcen,df=df,nfreq=nfreq) for v in objective_regions ]
        self.extra_cells        = [ DFTCell(region=v,fcen=fcen,df=df,nfreq=nfreq) for v in extra_regions ] if args.full_dfts else []
        self.design_cell        = DFTCell(region=design_region,fcen=fcen,df=df,nfreq=nfreq)
        self.dft_cells          = self.objective_cells + self.extra_cells + [self.design_cell]

        self.obj_func = ObjectiveFunction(self.objective_cells, f_str=f_str)

        self.beta_vector = self.init_beta_vector()

        self.outfile = args.outfile if args.outfile else "OptimizationProblem.out"

        adjoint_options['dft_reltol']   = args.dft_reltol
        adjoint_options['dft_timeout']  = args.dft_timeout
        adjoint_options['dft_interval'] = args.dft_interval
        adjoint_options['visualize']    = args.visualize
        adjoint_options['verbosity']    =      'verbose' if args.verbose    \
                                          else 'concise' if args.concise    \
                                          else adjoint_options['verbosity']

    ######################################################################
    # constructor helper method that initializes the command-line parser
    #  with general-purpose (problem-independent) arguments
    ######################################################################
    def init_args(self):
        parser = argparse.ArgumentParser()

        #--------------------------------------------------
        # parameters affecting geometry and sources
        #--------------------------------------------------
        parser.add_argument('--res',         type=float, default=20,      help='resolution')
        parser.add_argument('--dpml',        type=float, default=-1.0,    help='PML thickness (-1 --> autodetermined)')
        parser.add_argument('--fcen',        type=float, default=0.5,     help='center frequency')
        parser.add_argument('--df',          type=float, default=0.25,    help='frequency width')
        parser.add_argument('--source_mode', type=int,   default=1,       help='mode index of eigenmode source')

        #--------------------------------------------------
        # initial values for basis-function coefficients
        #--------------------------------------------------
        parser.add_argument('--betafile',  type=str,   default='',       help='file of expansion coefficients')
        parser.add_argument('--beta', nargs=2, default=[], action='append',  help='set value of expansion coefficient')

        #--------------------------------------------------
        # options describing the calculation to be done
        #--------------------------------------------------
        # do a single calculation of the objective function, optionally with gradient
        parser.add_argument('--eval_objective', dest='eval_objective',  action='store_true', help='evaluate objective function value')
        parser.add_argument('--eval_gradient',  dest='eval_gradient',   action='store_true', help='evaluate objective function value and gradient')

        # compute finite-difference approximation to derivative for test purposes
        parser.add_argument('--fd_order',      type=int,   default=0,        help='finite-difference order (0,1,2)')
        parser.add_argument('--fd_index',      type=int,   default=0,        help='index of differentiation variable')
        parser.add_argument('--fd_delta',      type=float, default=1.0e-2,   help='relative finite-difference delta')

        # run the full iterative optimization
        parser.add_argument('--optimize',       dest='optimize',       help='perform automated design optimization')

        #--------------------------------------------------
        # flags affecting the simulation run
        #--------------------------------------------------
        parser.add_argument('--nfreq',          type=int,              default=1,           help='number of output frequencies')
        parser.add_argument('--full_dfts',      dest='full_dfts',      action='store_true', help='compute DFT fields over full volume')
        parser.add_argument('--complex_fields', dest='complex_fields', action='store_true', help='force complex fields')
        parser.add_argument('--outfile',        type=str,   default=None,    help='output file name')

        #--------------------------------------------------
        # flags configuring adjoint-solver options
        #--------------------------------------------------l
        parser.add_argument('--dft_reltol',   type=float, default=adjoint_options['dft_reltol'],   help='convergence threshold for end of timestepping')
        parser.add_argument('--dft_timeout',  type=float, default=adjoint_options['dft_timeout'],  help='max runtime as a multiple of last_source_time')
        parser.add_argument('--dft_interval', type=float, default=adjoint_options['dft_interval'], help='meep time between DFT convergence checks')
        parser.add_argument('--visualize',    dest='visualize', action='store_true', help='produce visualization graphics')
        parser.add_argument('--verbose',      dest='verbose',   action='store_true', help='produce more output')
        parser.add_argument('--concise',      dest='concise',   action='store_true', help='produce less output')

        return parser

    ######################################################################
    # constructor helper function that interprets command-line arguments
    # to initialize vector of basis expansion coefficients
    ######################################################################
    def init_beta_vector(self):

        # default: beta[0] (constant term) = 1, all other coefficients=0
        nbeta=len(self.basis())
        beta_vector = np.zeros(nbeta)
        beta_vector[0]=1.0

        # try to interpret file in the form
        #    beta_0 \n beta_1 \n ... (if single column)
        # or
        #    i1	 beta_i1 \n i2 beta_i2 \n  (if two columns)
        if self.args.betafile:
            fb = np.loadtxt(self.args.betafile)   # 'file beta'
            if np.ndim(fb)==1:
                indices,vals = range(len(fb)), fb
            elif np.ndim(fb)==2 and np.shape(fb)[1] == 2:
                indices,vals = fb[:,0], fb[:,1]
            else:
                raise ValueError("{}: invalid file format".format(args.betafile))
            for i,v in zip(indices,vals): beta_vector[i]=v

        # parse arguments of the form --beta index value
        for ivpair in self.args.beta:     # loop over (index,value) pairs in args.beta
            beta_vector[int(ivpair[0])]=ivpair[1]

        # this step ensures that epsilon(x) never falls below 1.0
        eps_min=np.sum(np.abs(beta_vector))
        if eps_min<1.0:
            beta_vector[0]+=eps_min+1.0
        return beta_vector

    ######################################################################
    ######################################################################
    ######################################################################
    def optimize(self):
        print("Running optimization...")
        sys.exit(0)

    ######################################################################
    # do a single-point calculation of the objective function, plus
    # possibly its gradient (via the adjoint method) and/or its
    # derivative w.r.t. a single design variable via finite-differencing
    ######################################################################
    def single_point_calculation(self):

        args=self.args

        sim=self.create_sim(self.beta_vector)
        if adjoint_options['visualize']:
            fig=plt.figure(1)
            plt.clf()
            visualize_sim(sim)

        if args.eval_gradient:
           F,gradF=get_objective_and_gradient(sim,self.obj_func,
                                              self.dft_cells,self.basis)
        else:
           F = get_objective(sim,self.obj_func,self.dft_cells)
           gradF = 0.0*self.beta_vector

        FQ = [F] + self.obj_func.qvalues
        FQ_names = ['F'] + self.obj_func.qnames

        ##################################################
        # finite-difference derivatives
        ##################################################
        if args.fd_order > 0:
            delta_beta=args.fd_delta*np.abs(self.beta_vector[args.fd_index])
            if delta_beta==0.0:
                delta_beta=args.fd_delta
            beta_hat=0.0*self.beta_vector; beta_hat[args.fd_index]=1;
            sim  = self.create_sim(self.beta_vector + delta_beta*beta_hat)
            FP   = get_objective(sim,self.obj_func,self.dft_cells)
            FQP  = [FP] + self.obj_func.qvalues
            d1FQ = (FQP-FQ)/delta_beta

            if args.fd_order > 1:
                sim  = self.create_sim(self.beta_vector - delta_beta*beta_hat)
                FM   = get_objective(sim,self.obj_func,self.dft_cells)
                FQM  = [FM] + self.obj_func.qvalues
                d2FQ = (FQP-FQM)/(2.0*delta_beta)

        #--------------------------------------------------
        #- write results to console and output file
        #--------------------------------------------------
        f=sys.stdout
        f.write("** fcen, df, res, source_mode= %g, %g, %g %i \n" % (args.fcen,args.df,args.res,args.source_mode))
        for n,beta in enumerate(self.beta_vector):
            if self.beta_vector[n]!=0.0:
                f.write("beta[%i]=%e\n" % (n,self.beta_vector[n]))
        for n,x in enumerate(FQ):
            f.write("%s: {%+.4e,%+.4e} " % (FQ_names[n], np.real(x), np.imag(x)))
            if args.fd_order>0:
                f.write("d1: {%+.4e %+.4e} " % (np.real(d1FQ[n]), np.imag(d1FQ[n])))
            if args.fd_order>1:
                f.write("d2: {%+.4e %+.4e} " % (np.real(d2FQ[n]), np.imag(d2FQ[n])))
            f.write('\n')
        if args.eval_gradient:
            for n,x in enumerate(gradF):
                f.write('dF/db%02i = {%+.4e,%+.4e}\n' % (n,np.real(gradF[n]),np.imag(gradF[n])))
        f.write('\n\n')

        f=open(self.outfile,"a")
        f.write("%e %e %e %i " % (args.fcen,args.df,args.res,args.source_mode))
        [f.write("%e " % beta) for beta in self.beta_vector]
        for n,x in enumerate(FQ):
            f.write("%e %e " % (np.real(x), np.imag(x)))
            if args.fd_order>0:
                f.write("%e %e " % (np.real(d1FQ[n]), np.imag(d1FQ[n])))
            if args.fd_order>1:
                f.write("%e %e " % (np.real(d2FQ[n]), np.imag(d2FQ[n])))
        if args.eval_gradient:
            [f.write("%e %e " % (np.real(g), np.imag(g))) for g in gradF]
        f.write("\n")
        f.close()

    ######################################################################
    # 'run' class method of OptimizationProblem branches off among a
    # variety of possible computations based on command-line options
    ######################################################################
    def run(self):

        #--------------------------------------------------------------
        # run iterative design optimization
        #--------------------------------------------------------------
        args=self.args
        if args.optimize:
            self.optimize()
        #--------------------------------------------------------------
        # calculate the objective function and possibly derivatives at
        # just the single given input point
        #--------------------------------------------------------------
        elif args.eval_gradient or args.eval_objective or args.fd_order>0:
            self.single_point_calculation()
        #--------------------------------------------------------------
        # just plot a diagram of the geometry with the given design variables
        #--------------------------------------------------------------
        else:
            print("visualize_sim()")
