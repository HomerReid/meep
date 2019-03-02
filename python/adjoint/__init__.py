""" Adjoint-based sensitivity-analysis submodule for the MEEP python module.

Docmentation:
http://https://meep.readthedocs.io/en/latest/Python_Tutorials/AdjointSolver.md

The code is structured as follows:

    __init__.py:   	     formal definitions

    OptimizationProblem.py:  abstract base class from which user-supplied
                             classes describing specific optimization
                             geometries should inherit; implements a high-level
                             notion of a differentiable objective function
                             depending on one or more input variables, which we
                             call "objective quantities."

    ObjectiveFunction.py:    lower-level routines for carrying out MEEP timestepping
                             calculations to compute objective quantities and their
                             adjoint derivatives.

    Basis.py:                general support for spatially-varying permittivity
                             functions described by expansions in user-defined sets
                             of basis functions, plus implementations of simple basis sets
                             for some common cases

    Visualization.py:        routines for visualizing MEEP input geometries
                             and computational results.

Minimal example:

import meep as mp
import mp.adjoint as adj

class MyOptProblem(adj.OptimizationProblem):

    ##################################################
    # define problem-specific command-line options and
    # problem-specific default values for general
    # command-line optionsj
    ##################################################
    def add_args(self, parser):
        parser.add_argument('--radius', type=float, default=1.0, help='radius')
        parser.add_argument('--depth',  type=float, default=0.5, help='depth')

        parser.set_defaults(fcen=0.5)
        parser.set_defaults(df=0.1)


    ##################################################
    # routine called just once, before any calculations
    # or optimization, to define aspects of the simulation
    # geometry that remain constant independent of the
    # values of design variables.
    ##################################################
    def initialize(self,args):
        ...
        ...
        ...
        return f_expr, objective_regions, design_regions, extra_regions, basis

    ##################################################
    # routine called many times over the course of an
    # iterative simulation process to instantiate a
    # MEEP simulation for a given set of values of
    # the design variables
    ##################################################
    def create_sim(self,beta_vector,vacuum=False):
        ...
        ...
        ...
        return sim

"""

__all__ = [ 'OptimizationProblem', 'EHTransverse' ]

from .OptimizationProblem import OptimizationProblem

from .ObjectiveFunction import (EHTransverse, Exyz, Hxyz, EHxyz, xHat, yHat,
                                zHat, Origin, GridInfo, ObjectiveFunction,
                                DFTCell, AdjointOptions, Abs2, RelDiff)

from .Basis import ParameterizedDielectric, PlaneWaveBasis, FourierLegendreBasis

from .Visualize import VisualizeSim, PlotOptions
