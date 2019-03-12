import sys
import os
import argparse
import pickle
import datetime

import numpy as np

import matplotlib.pyplot as plt

import meep as mp
from . import ObjectiveFunction
from . import Visualization
from . import Basis

from ObjectiveFunction import (adjoint_options, xHat, yHat, zHat, origin,
                               EHTransverse, Exyz, Hxyz, EHxyz, GridInfo,
                               abs2, unit_vector, rel_diff, FluxLine,
                               DFTCell, ObjectiveFunction, AdjointSolver)

from Visualization import visualize_sim, plot_options, AdjointVisualizer

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

        # define command-line arguments
        parser = self.init_args()       # general args
        self.add_args(parser)           # problem-specific arguments
        self.args = args = parser.parse_args()

        # parse arguments and do problem-specific initialization first...
        fstr, objective_regions, extra_regions, design_region, self.basis \
            = self.init_problem(self.args)

        # ...and now do some general initialization
        fcen, df, nfreq         = args.fcen, args.df, args.nfreq
        #self.grid  = FDGrid(size=self.cell_size, res=self.args.res)
        self.objective_cells    = [ DFTCell(region=v, fcen=fcen, df=df, nfreq=nfreq) for v in objective_regions ]
        self.extra_cells        = [ DFTCell(region=v, fcen=fcen, df=df, nfreq=nfreq) for v in extra_regions ] if args.full_dfts else []
        self.design_cell        = DFTCell(region=design_region, fcen=fcen, df=df, nfreq=nfreq, name='design_fields')
        self.dft_cells          = self.objective_cells + self.extra_cells + [self.design_cell]

        self.obj_func           = ObjectiveFunction(fstr=fstr)

        self.dimension          = len(self.basis()) # dimension of optimization problem = number of design variables
        self.beta_vector        = self.init_beta_vector()

        self.sim                = None

        self.filebase           = args.filebase if args.filebase else self.__class__.__name__

        adjoint_options['dft_reltol']   = args.dft_reltol
        adjoint_options['dft_timeout']  = args.dft_timeout
        adjoint_options['dft_interval'] = args.dft_interval
        adjoint_options['verbosity']    = 'verbose' if args.verbose         \
                                          else 'concise' if args.concise    \
                                          else adjoint_options['verbosity']
        adjoint_options['logfile']      = args.logfile

        self.vis = AdjointVisualizer() if args.visualize else None
        adjoint_options['animate_components'] = args.animate_component
        adjoint_options['animate_interval']   = args.animate_interval

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
        parser.add_argument('--eval_objective',  dest='eval_objective',  action='store_true', help='evaluate objective function value')
        parser.add_argument('--eval_gradient',   dest='eval_gradient',   action='store_true', help='evaluate objective function value and gradient')
        parser.add_argument('--gradient_qname',  type=str, default=None, help='name of objective quantity to differentiate via adjoint method')

        # compute finite-difference approximation to derivative for test purposes
        parser.add_argument('--fd_order',      type=int,   default=0,        help='finite-difference order (0,1,2)')
        parser.add_argument('--fd_index',      default=[], action='append',  help='index of differentiation variable')
        parser.add_argument('--fd_rel_delta',  type=float, default=1.0e-2,   help='relative finite-difference delta')

        # run the full iterative optimization
        parser.add_argument('--optimize',       dest='optimize',       help='perform automated design optimization')

        #--------------------------------------------------
        # flags affecting the simulation run
        #--------------------------------------------------
        parser.add_argument('--nfreq',          type=int,              default=1,           help='number of output frequencies')
        parser.add_argument('--full_dfts',      dest='full_dfts',      action='store_true', help='compute DFT fields over full volume')
        parser.add_argument('--complex_fields', dest='complex_fields', action='store_true', help='force complex fields')
        parser.add_argument('--filebase',       type=str,              default=self.__class__.__name__, help='base name of output files')

        #--------------------------------------------------
        # flags configuring adjoint-solver options
        #--------------------------------------------------l
        parser.add_argument('--dft_reltol',   type=float, default=adjoint_options['dft_reltol'],   help='convergence threshold for end of timestepping')
        parser.add_argument('--dft_timeout',  type=float, default=adjoint_options['dft_timeout'],  help='max runtime in units of last_source_time')
        parser.add_argument('--dft_interval', type=float, default=adjoint_options['dft_interval'], help='meep time DFT convergence checks in units of last_source_time')
        parser.add_argument('--verbose',      dest='verbose',   action='store_true', help='produce more output')
        parser.add_argument('--concise',      dest='concise',   action='store_true', help='produce less output')
        parser.add_argument('--visualize',    dest='visualize', action='store_true', help='produce visualization graphics')
        parser.add_argument('--logfile',        type=str,   default=None,    help='log file name')
        parser.add_argument('--animate_component', action='append', help='plot time-domain field component')
        parser.add_argument('--animate_interval', type=float, default=1.0, help='meep time between animation frames')

        return parser

    ######################################################################
    # constructor helper method to process command-line arguments for
    # initializing the vector of design variables
    ######################################################################
    def init_beta_vector(self):

        # default: beta_0 (constant term) = 1, all other coefficients=0
        beta_vector=unit_vector(0,self.dimension)

        # if a --betafile was specified, try to parse it in the form
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
                raise ValueError("{}: invalid file format".format(self.args.betafile))
            for i,v in zip(indices,vals): beta_vector[i]=v

        # parse arguments of the form --beta index value
        for ivpair in self.args.beta:     # loop over (index,value) pairs
            beta_vector[int(ivpair[0])]=ivpair[1]

        # this step ensures that epsilon(x) never falls below 1.0, assuming
        # (a) the 0th basis function is the constant 1
        # (b) all other basis functions (b_1, ... b_{D-1}) take values in [-1:1]
        # (c) coefficients (\beta_1, ..., \beta_{D-1}) are nonnegative.
        eps_min=beta_vector[0] - np.sum(beta_vector[1:])
        if eps_min<1.0:
            beta_vector[0] += 1.0-eps_min
        return beta_vector

    ######################################################################
    # the 'run' class method of OptimizationProblem, which is where control
    # passes after __init__ if we were executed as a script, looks at
    # command-line flags to determine whether to (a) launch a full
    # iterative optimization of the design variables or (b) study
    # the geometry for just the one given set of design variables
    ######################################################################
    def run(self):

        #--------------------------------------------------------------
        # if --optimize is present, run iterative design optimization
        #--------------------------------------------------------------
        args=self.args
        if args.optimize:
            self.optimize()
            return 0

        #--------------------------------------------------------------
        # if no computation was requested, just plot the geometry
        #--------------------------------------------------------------
        if not(args.eval_gradient or args.eval_objective or args.fd_order>0):
            self.sim = self.create_sim(self.beta_vector)
            if self.vis:
                self.vis.update(self.sim,'Geometry')
            return 0

        #--------------------------------------------------------------
        #--------------------------------------------------------------
        #--------------------------------------------------------------
        fq,gradf=self.eval_fq(self.beta_vector, need_gradient=args.eval_gradient, vis=self.vis)

        fqnames=['f'] + self.obj_func.qnames
        self.output([],[],actions=['begin'])
        self.output(['res','fcen','df','source_mode'], [args.res,args.fcen,args.df,args.source_mode])
        [self.output(['beta'+str(n)],[beta]) for (n,beta) in enumerate(self.beta_vector)]
        [self.output([fqn], [fqv]) for (fqn,fqv) in zip(fqnames,fq)]
        if args.eval_gradient:
            [self.output( ['df/db'+str(n)], [g]) for n,g in enumerate(gradf)]

        #--------------------------------------------------------------
        # calculate the objective function value, optionally its gradient,
        # and optionally the finite-difference approximation to the
        # derivative of a single objective quantity
        #--------------------------------------------------------------
        indices = [] if args.fd_order==0 else [int(index) for index in args.fd_index]
        for index in indices:
            dbeta    = args.fd_rel_delta*(1.0 if self.beta_vector[index]==0 else np.abs(self.beta_vector[index]))
            beta_hat = unit_vector(index,self.dimension)
            fqp, _ = self.eval_fq(self.beta_vector + dbeta*beta_hat)
            d1fq   = (fqp - fq) / dbeta
            if args.fd_order==2:
                fqm, _ = self.eval_fq(self.beta_vector - dbeta*beta_hat)
                d2fq   = (fqp - fqm) / (2.0*dbeta)
            for i,fqn in enumerate(fqnames):
                nlist = ['d{}/db{}__O{}__'.format(fqn,index,ord+1) for ord in range(args.fd_order)]
                vlist = [ d1fq[i] ] + ([d2fq[i]] if args.fd_order==2 else[])
                [self.output(nlist, vlist)]

        self.output([],[],actions=['end'])

        #f = open('store.pckl', 'wb')
        # pickle.dump(obj, f)
        #f.close()
        #f = open('store.pckl', 'rb')
        #obj = pickle.load(f)
        #f.close()

    ######################################################################
    ######################################################################
    ######################################################################
    def output(self, names, values, actions=[]):

        if 'begin' in actions:
            self.nout = 1 # running index of output quantity
            self.legend  = open(self.filebase + '.legend','w')
            self.digest  = open(self.filebase + '.digest','a')
            self.outfile = open(self.filebase + '.out','a')
            dt=datetime.datetime.now().strftime('%D::%T ')
            for f in [sys.stdout,self.digest]:
                f.write('\n\n** {} ran {} \n'.format(dt,self.__class__.__name__))
                f.write('** with args {}'.format(' '.join(sys.argv[1:])))
                f.write('\n**')

        [self.legend.write('{} {}\n'.format(self.nout+i,n)) for i,n in enumerate(names)]
        self.nout+=len(names)
        for v in values:
            self.outfile.write(' {:+.6e} {:+.6e}  '.format(np.real(v),np.imag(v)))
        for f in [sys.stdout,self.digest]:
            f.write('{:20s}: '.format(','.join(names)) )
            for v in values:
                f.write('({:+.6e},{:+.6e}) '.format(np.real(v),np.imag(v)))
            f.write('\n')

        if 'end' in actions:
            self.outfile.write('\n')
            self.outfile.close()
            self.digest.close()
            self.legend.close()

    ######################################################################
    ######################################################################
    ######################################################################
    def optimize(self):
        print("Running optimization...")
        sys.exit(0)

    ######################################################################
    # compute the objective function, plus possibly its gradient (via
    # adjoints), at a single point in design space
    ######################################################################
    def eval_objective(self, beta_vector, need_gradient=False, vis=None):

        sim = self.create_sim(beta_vector)
        solver = AdjointSolver(self.obj_func, self.dft_cells, self.basis, sim, vis=vis)
        self.sim=sim # save a copy for debugging; not strictly necessary
        return solver.solve(need_gradient=need_gradient)

    ##################################################
    ##################################################
    ##################################################
    def eval_fq(self, beta_vector, need_gradient=False, vis=None):
        f,gradf = self.eval_objective(beta_vector, need_gradient=need_gradient, vis=vis)
        fq = np.array( [f] + list(self.obj_func.qvals) )
        return fq, gradf
