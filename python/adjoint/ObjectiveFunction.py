#####################################################
# ObjectiveFunction.py --- routines for evaluating
# objective quantities and objective functions for
# adjoint-based optimization in meep
#####################################################
import sys
import datetime
import re
import copy
import sympy
from collections import namedtuple
import numpy as np
import meep as mp

import matplotlib.pyplot as plt

from .Basis import project_basis
from .Visualize import visualize_sim, plot_options

##################################################
# some convenient constants and typedefs
##################################################
EHTransverse=[ [mp.Ey, mp.Ez, mp.Hy, mp.Hz],
               [mp.Ez, mp.Ex, mp.Hz, mp.Hx],
               [mp.Ex, mp.Ey, mp.Hx, mp.Hy] ]
Exyz=[mp.Ex, mp.Ey, mp.Ez]
Hxyz=[mp.Hx, mp.Hy, mp.Hz]
EHxyz=Exyz+Hxyz

xhat=mp.Vector3(1.0,0.0,0.0)
yhat=mp.Vector3(0.0,1.0,0.0)
zhat=mp.Vector3(0.0,0.0,1.0)
origin=mp.Vector3()

# FDGrid stores info on the full Yee grid in a MEEP simulation
FDGrid = namedtuple('FDGrid', ['size', 'res'])

VERBOSE=2
STANDARD=1
CONCISE=0

######################################################################
# various options affecting the adjoint solver, user-tweakable via
# command-line arguments or set_adjoint_option()
######################################################################
adjoint_options={
 'dft_reltol':    1.0e-6,
 'dft_timeout':   10.0,
 'dft_interval':  20.0,
 'verbosity':     STANDARD,
 'visualize':     False
}

##################################################
# miscellaneous utility routines
##################################################
def rel_diff(a,b):
    return 0.0 if a==0.0 and b==0.0 else np.abs(a-b)/np.mean([np.abs(a),np.abs(b)])


def abs2(z):
    return np.real(np.conj(z)*z)


######################################################################
# DFTCell is an improved data structure for working with frequency-domain
# field components in MEEP calculations. It consolidates and replaces the
# zoo of 8 quasi-redundant DFT-related data structures in core meep (namely:
# DftFlux, DftFields, DftNear2Far, DftForce, DftLdos, DftObj, FluxRegion,
# FieldsRegion) and has a different relationship to individual MEEP
# simulations described by instances of mp.simulation().
#
# In a nutshell, the core constituents of a DFT cell are the three metadata
# fields that define the set of frequency-domain field amplitudes tabulated
# by the cell: a grid subvolume (including the associated 'xyzw' metadata),
# a set of field components, and a set of frequencies. These fields are passed
# to the DFTCell constructor and do not change over the lifetime of the DFTCell,
# which will generally encompass the lifetimes of several mp.simulation()
# instances.
#
# On the other hand, the frequency-domain field-amplitude data arrays produced by
# DFT calculations are considered less intrinsic: the DFT cell may have *no* such
# data (as when a calculation is first initiated), or may have multiple sets of
# data arrays resulting from multiple different timestepping runs. These multiple
# data sets may correspond, for example, to timestepping runs excited by different
# sources (such as the forward and adjoint sources in an adjoint-based value-and-gradient
# calculation) and/or to runs in the full and 'vacuum' versions of a geometry,
# where the latter is the 'bare' version of a geometry (with scatterers and obstacles
# removed) that one runs to tabulate incident-field profiles.
#
# Although not strictly related to DFT calculations, DFTCells describing flux-monitor
# regions also know how to compute and cache arrays of eigenmode field amplitudes.
#
# We use the following semantic conventions for the names of variables representing
# multidimensional arrays of frequency-domain field amplitudes.
#
#   -- EH refers to a (1D, 2D, or 3D) array of frequency-domain field amplitudes
#      for a single field component at a single frequency.
#
#   -- An 'EHList' is a one-dimensional list of EH arrays, one for each component in the
#      cell. Thus e.g. EHList[2] = array slice of amplitudes for component cEH[2], all at
#      a single frequency.
#
#   -- An 'EHSet' is a one-dimensional list of EHLists, one for each frequency in a the DFTCell.
#
#   -- Arrays of eigenmode field amplitudes are named similarly with the substitution "EH" -> "eh"
#
# Note: for now, all arrays are stored in memory. For large calculations
# with many DFT frequencies this may become impractical. TODO: implement disk caching.
######################################################################
class DFTCell(object):

    ######################################################################
    ######################################################################
    ######################################################################
    def __init__(self, FDGrid=None, region=None, center=origin, size=None,
                       cEH=None, fcen=None, df=0, nfreq=1):

        if region is not None:
            self.center, self.size, self.region = region.center, region.size, region
        elif size is not None:
            self.center, self.size, self.region = center, size, mp.Volume(center=center, size=size)
        else:
            self.center, self.size, self.region = origin, FDGrid.size, mp.Volume(center=center, size=size)

        self.nhat     = region.direction if hasattr(region,'direction') else None
        self.type     = 'flux' if self.nhat is not None else 'fields' # TODO extend to other cases
        self.cEH      =      cEH if cEH is not None                                  \
                        else EHTransverse[self.nhat - mp.X] if self.nhat is not None \
                        else EHxyz
        self.fcen     = fcen
        self.df       = df if nfreq>1 else 0.0
        self.nfreq    = nfreq
        self.freqs    = [fcen] if nfreq==0 else np.linspace(fcen-0.5*df, fcen+0.5*df, nfreq)

        self.sim      = None  # mp.simulation for current simulation
        self.dft_obj  = None  # meep DFT object for current simulation
        self.labels   = []    # labels[nsim] = name of #nsimth run for which we have data
        self.EHSets   = []    # EHSets[nsim][nf][nc] = array slice for component #nc at freq #nf for run #nsim

        self.last_update_time = 0.0

        self.eigencache = {}  # dict of eigenmode field EHLists, indexed by (mode,freq) pairs


        # FIXME At present the 'xyzw' metadata cannot be computed until a mp.simulation / meep::fields
        #       object has been created, but in fact the metadata only depend on the FDGrid
        #       (resolution and extents of the computational lattice) and are independent
        #       of the specific material geometry and source configuration of any particular
        #       'fields' instance or simulation. In keeping with the spirit of 'DFTCell' it should
        #       be possible to compute the metadata once and for all right here before any mp.simulation()
        #       or meep::fields is created, but that will require some code refactoring. For the time being
        #       we punt on this until later, after a fields object has been created.
        self.xyzw, self.slice_dims  = None, None

    ######################################################################
    # initialize and return an empty 'EHSet'
    ######################################################################
    def NewEHSet(self):
        return [ [ 0.0j*np.zeros(self.slice_dims) for c in self.cEH ] for f in self.freqs ]

    ######################################################################
    # Look up and return the EHList for a given frequency in a given run
    ######################################################################
    def RetrieveEHList(self,label,nf=0):
        if label not in self.labels or nf>len(self.EHSets[self.labels.index(label)]):
            ValueError('requested non-existent data for run={}, nf={}'.format(label,nf))
        return self.EHSets[self.labels.index(label)][nf]

    ######################################################################
    ######################################################################
    ######################################################################
    def SubtractIncidentFields(self,EHTList,nf=0):
        EHIList = self.RetrieveEHList('incident',nf)
        for nc in range(len(cEH)):
            EHTList[nc] -= EHIList[nc]

    ######################################################################
    # 'register' the cell with a MEEP timestepping simulation to request
    # computation of frequency-domain fields
    ######################################################################
    def Register(self, sim, label):
        # take the opportunity to fill in the metadata if not done yet; #FIXME to be removed as discussed above
        if self.xyzw is None:
            self.xyzw  = sim.get_dft_array_metadata(center=self.center, size=self.size)
            self.slice_dims = np.shape(self.xyzw[3])

        self.sim     = sim
        self.dft_obj =      sim.add_flux(self.fcen,self.df,self.nfreq,self.region) if self.type=='flux'   \
                       else sim.add_dft_fields(self.cEH, self.fcen, self.df, self.nfreq, where=self.region)
        self.labels.append(label)
        self.EHSets.append( self.NewEHSet() )
        self.last_update_time = 0.0

    ######################################################################
    # This is like mp.get_dft_array(), but 'zero-padded:' when the DFT cell
    # does not have data for the requested component (perhaps because it
    # vanishes identically by symmetry), this routine returns an array of the
    # expected dimensions with all zero entries, instead of a rank-0 array that
    # prints out as a single preposterously large or small floating-point
    # number, which is the not-very-user-friendly behavior of get_dft_array().
    ######################################################################
    def GetEH(self, c, nf=0):
        EH = self.sim.get_dft_array(self.dft_obj, c, nf)
        return EH if np.ndim(EH)>0 else 0.0j*np.zeros(self.slice_dims)

    ######################################################################
    ######################################################################
    ######################################################################
    def UpdateEHSets(self):
        if self.last_update_time < self.sim.round_time():
            self.last_update_time=self.sim.round_time()
            self.EHSets = [ [ self.GetEH(c, nf) for c in self.cEH ] for nf in range(self.nfreq) ]

    ######################################################################
    # Return an ehList describing the fields of eigenmode #mode at freq #nf.
    ######################################################################
    def GetEigenfields(self, mode, nf=0):

        # look for data in cache
        freq=self.freqs[nf]
        tag='M{}.F{:.6f}'.format(mode,freq)
        if self.eigencache and tag in self.eigencache:
            return self.eigencache[tag]

        vol=mp.Volume(self.region.center,self.region.size)
        eigenmode=self.sim.get_eigenmode(freq, self.nhat, vol, mode, mp.Vector3())
        ehList=[0.0j*np.zeros(self.slice_dims) for c in self.cEH]
        x,y,z,w=self.xyzw[0], self.xyzw[1], self.xyzw[2], self.xyzw[3]
        xyz = [mp.Vector3(xx,yy,zz) for xx in x for yy in y for zz in z]
        for nc,c in enumerate(self.cEH):
            nxyz=0
            for n,ww in np.ndenumerate(w):
                ehList[nc][n]=eigenmode.amplitude(xyz[nxyz], c)
                nxyz+=1

        # store in cache before returning
        if self.eigencache is not None:
            print("Adding eigenfields for tag {}".format(tag))
            self.eigencache[tag]=ehList

        return ehList

    ##################################################
    # compute an objective quantity, i.e. an eigenmode
    # coefficient or a scattered or total power
    ##################################################
    def EvalQuantity(self, qtype, mode, nf=0):

        self.UpdateEHSets()
        w=self.xyzw[3]

        EHList=self.EHSets[-1][nf]    # use field data from most recent simulation
        if qtype.islower():           # need scattered fields?
            self.SubtractIncidentFields(EHList,nf)

        if qtype in 'sS':
            return 0.25*np.real(np.sum(w*( np.conj(EHList[0])*EHList[3] - np.conj(EHList[1])*EHList[2]) ))
        elif qtype in 'PM':
            ehList=self.GetEigenfields(mode, nf)  # EHList of eigenmode fields
            eH = np.sum( w*(np.conj(ehList[0])*EHList[3] - np.conj(ehList[1])*EHList[2]) )
            hE = np.sum( w*(np.conj(ehList[3])*EHList[0] - np.conj(ehList[2])*EHList[1]) )
            sign=1.0 if qtype=='P' else -1.0
            return (eH + sign*hE)/2.0
        else: # TODO: support other types of objectives quantities?
            ValueError('unsupported quantity type')


#def flux_line(x0, y0, length, dir):
#    size=length*(xhat if dir is mp.Y else yhat)
#    return mp.FluxRegion(center=mp.Vector3(x0,y0), size=size, direction=dir)



#########################################################
# ObjectiveFunction is a simple class for evaluating
# arbitrary user-specified objective functions; its data are
# (a) a mathematical expression for the function (a
#     character string in which the names of various
#     objective quantities like 'S_0' or 'P1_1' appear), and
# (b) the DFTCells needed to compute the objective quantities.
#########################################################
class ObjectiveFunction(object):

    ######################################################################
    ######################################################################
    ######################################################################
    def __init__(self, objective_cells, f_str='S0'):
        try:
            self.f_expr = sympy.sympify(f_str)
        except:
            raise ValueError("failed to parse function {}".format(f_str))
        self.qnames=[str(s) for s in self.f_expr.free_symbols]
        self.qnames.sort()
        self.qvalues=np.zeros(len(self.qnames))
        self.qdict={}
        self.objective_cells = objective_cells
        # sanity check names of all objective quantities
        for q in self.qnames:
            qtype,mode,ncell = ObjectiveFunction.unpack_quantity_name(q) # aborts on parse error
            if ncell >= len(objective_cells):
                raise ValueError("quantity {}: reference to non-existent objective cell".format(q))

    #########################################################
    # attempt to parse the name of an objective quantity like
    # 'P2_3' or 's_4'. On success, return triple
    # (type, eigenmode_index, region_index).
    #########################################################
    @staticmethod
    def unpack_quantity_name(qname):

        # mode coefficient: look for a match of the form 'P1_2' or 'F2_0'
        mode_match=re.search(r'([PM])([1-9]*)_([0-9])',qname)
        if mode_match is not None:
            return mode_match[1], int(mode_match[2]), int(mode_match[3])

        # total/scattered power: look for a match of the form 'S_1' or 's_2'
        flux_match=re.search(r'([Ss])_([0-9])',qname)
        if flux_match is not None:
            return flux_match[1], 0, int(flux_match[2])

        raise ValueError("invalid quantity name " + qname)


  ######################################################################
  # (re)compute the values of all objective quantities.
  ######################################################################
    def update_quantities(self, nf=0):
        for nq, qname in enumerate(self.qnames):
            qtype, mode, ncell = ObjectiveFunction.unpack_quantity_name(qname)
            qvalue=self.objective_cells[ncell].EvalQuantity(qtype, mode, nf=nf)
            self.qvalues[nq]=self.qdict[qname]=qvalue


  ######################################################################
  # evaluate the objective function
  ######################################################################
    def eval(self):
        self.update_quantities()
        val=self.f_expr.evalf(subs=self.qdict)
        return complex(val) if val.is_complex else float(val)


  ######################################################################
  # evaluate partial derivatives of the objective function (assumes
  # internally-stored quantity values are up to date, i.e. that no
  # timestepping has happened since the last call to update_quantities)
  ######################################################################
    def partial(self, qname):
        val=sympy.diff(f_expr,qname).evalf(subs=self.qdict)
        return complex(val) if val.is_complex else float(val)

#########################################################
# end of ObjectiveFunction class definition
#########################################################

# the default quantities to compute for a given dft_cell if no quantity
# is specified: total power flux (for a dft_flux cell) or total
# electromagnetic energy (for a dft_fields cell)
def default_qname(dft_cell):
    return 'S' if dft_cell_type(dft_cell)=='flux' else 'U'

def default_objective_function(objective_dft_cells):
    f_string=''
    for (n,cell) in enumerate(objective_dft_cells):
        f_string += ('+{}{}'.format(default_qname(cell),n))
    return ObjectiveFunction(f_string)

#########################################################
#########################################################
#########################################################
def eval_gradient(sim, DFTCells, basis):

     gradient=0.0j*np.zeros(len(basis()))
     cell=DFTCells[-1] # design cell
     EHForward=cell.RetrieveEHList('forward')
     cell.UpdateEHSets()
     EHAdjoint=cell.RetrieveEHList('adjoint')
     f=0.0j*np.ones(cell.slice_dims)
     for nc,c in enumerate(cell.cEH):
         if c not in Exyz:
             continue
         f+=EHForward[nc]*EHAdjoint[nc]
     return project_basis(basis,cell.xyzw,f)

#########################################################
# Given a list of DFT cells, continue timestepping until the
# frequency-domain fields have stopped changing within the
# specified relative tolerance or the time limit is reached.
#########################################################
def run_until_dfts_converged(sim, DFTCells, ObjFunc, basis=None):

    reltol=adjoint_options['dft_reltol']
    max_time=adjoint_options['dft_timeout']*sim.fields.last_source_time()
    verbose=adjoint_options['verbosity']>STANDARD
    check_interval=adjoint_options['dft_interval']

    sim.run(until=sim.fields.last_source_time())
    next_check_time = sim.round_time() + check_interval
    if basis:
        Q       = eval_gradient(sim, DFTCells, basis)
        Q_names = ['b{}'.format(n) for n in range(len(basis()))]
    else:
        Q       = np.array( [ObjFunc.eval()] + ObjFunc.qvalues )
        Q_names = ['F'] + ObjFunc.qnames
    last_Q = Q
    while True:
        sim.run(until=next_check_time)
        next_check_time += check_interval

        if basis:
            Q = eval_gradient(sim, DFTCells, basis)
        else:
            Q = np.array( [ObjFunc.eval()] + ObjFunc.qvalues )
        delta = np.abs(Q-last_Q)
        rel_delta=[rel_diff(x,last_x) for x,last_x in zip(Q,last_Q)]
        max_rel_delta=max(rel_delta)
        last_Q=Q

        if mp.am_master() and verbose:
           dt=datetime.datetime.now().strftime("%T ")
           sys.stdout.write("\n**\n**%s %5i %.0e\n**\n" % (dt,sim.round_time(),max_rel_delta))
           [sys.stdout.write("{:10s}: {:+.4e}({:.1e})\n".format(n,q,e)) for n,q,e in zip(Q_names,Q,rel_delta)]
           sys.stdout.write("]\n\n")

        if max_rel_delta<=reltol or sim.round_time()>=max_time:
            return Q

##############################################################
##############################################################
##############################################################
def place_adjoint_sources(sim, envelope, qname, DFTCells):

   freq     = envelope.frequency
   omega    = 2.0*np.pi*freq
   factor   = 2.0j*omega
   if callable(getattr(envelope, "fourier_transform", None)):
       factor /= envelope.fourier_transform(omega)

   nf=0
   (qtype,mode,ncell)=ObjectiveFunction.unpack_quantity_name(qname)
   cell=DFTCells[ncell]
   if mode==0:
       EHList=cell.RetrieveEHList('forward',nf)
   else:
       EHList=cell.GetEigenfields(mode,nf)

   cEH=cell.cEH
   (x,y,z,w)=cell.xyzw[0],cell.xyzw[1],cell.xyzw[2],cell.xyzw[3]
   shape=[np.shape(q)[0] for q in [x,y,z]]

   if qtype in 'PM':
       sign = 1.0 if qtype=='P' else -1.0
       signs=[+1.0,-1.0,+1.0*sign,-1.0*sign]
       sim.sources+=[mp.Source(envelope, cEH[3-nc],
                               cell.center, cell.size,
                               amplitude=signs[nc]*factor,
                               amp_data=np.reshape(np.conj(EHList[nc]),shape)
                              ) for nc in range(4)]


########################################################
########################################################
########################################################
def get_objective_and_gradient(sim, ObjFunc, DFTCells,
                               basis=None, qname=None):

    #--------------------------------------------------------------
    #- forward run
    #--------------------------------------------------------------
    for cell in DFTCells:
        cell.Register(sim, 'forward')
    sim.init_sim()
    FQ=run_until_dfts_converged(sim, DFTCells, ObjFunc=ObjFunc)
    F=FQ[0]

    DFTCells[-1].UpdateEHSets()

    if adjoint_options['visualize']:
        fig=plt.figure(2)
        plt.clf()
        visualize_sim(sim)

    if basis is None:
        return F

    # --------------------------------------------------
    # adjoint run
    # --------------------------------------------------
    envelope=sim.sources[0].src
    sim.reset_meep()
    sim.change_sources([])
    qname=qname if qname else ObjFunc.qnames[0]
    place_adjoint_sources(sim,envelope,qname,DFTCells)
    import ipdb; ipdb.set_trace()
    for cell in DFTCells:
        cell.Register(sim, 'adjoint')
    sim.force_complex_fields=True
    sim.init_sim()
    gradient=run_until_dfts_converged(sim, DFTCells, ObjFunc, basis=basis)

    return F, gradient

#########################################################
#########################################################
#########################################################
def get_objective(sim, ObjFunc, DFTCells):
    return get_objective_and_gradient(sim, ObjFunc, DFTCells)
