####################################################
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

from . import ObjectiveFunction
from . import Visualization
from . import Basis

from .Basis import project_basis

##################################################
# some convenient constants and typedefs
##################################################
EHTransverse=[ [mp.Ey, mp.Ez, mp.Hy, mp.Hz],
               [mp.Ez, mp.Ex, mp.Hz, mp.Hx],
               [mp.Ex, mp.Ey, mp.Hx, mp.Hy] ]
Exyz=[mp.Ex, mp.Ey, mp.Ez]
Hxyz=[mp.Hx, mp.Hy, mp.Hz]
EHxyz=Exyz+Hxyz

xHat=mp.Vector3(1.0,0.0,0.0)
yHat=mp.Vector3(0.0,1.0,0.0)
zHat=mp.Vector3(0.0,0.0,1.0)
origin=mp.Vector3()

# GridInfo stores the extents and resolution of the full Yee grid in a MEEP
# simulation; this is the minimal information needed to compute array metadata.
GridInfo = namedtuple('GridInfo', ['size', 'res'])

######################################################################
# various options affecting the adjoint solver, user-tweakable via
# command-line arguments or set_adjoint_option()
######################################################################
adjoint_options={
 'dft_reltol':    1.0e-6,
 'dft_timeout':   10.0,
 'dft_interval':  20.0,
 'verbosity':     'default',
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
# Although not strictly related to DFT calculations, dft_cells describing flux-monitor
# regions also know how to compute and cache arrays of eigenmode field amplitudes.
#
# We use the following semantic conventions for choosing names for entities in the
# hierarchy of
#
#   -- At the most granular level, we have 1D, 2D, or 3D arrays (slices)
#      of frequency-domain amplitudes for a single field component at a
#      single frequency in a single calculation. For our purposes these
#      only ever arise as loop variables, which we typically call 'F' for field.
#
#   -- We often have occasion to refer to the full set of such slices
#      for all field components stored in the DFTCell, again at a single
#      frequency in a single simulation. Since this set typically includes
#      data for both E and H field components, we call it simply EH.
#      Thus, in general, EH[2] = data array for field component #2,
#      at a single frequency in a single simulation.
#
#   -- EHData refers to a collection (one-dimensional list) of EH arrays, one
#      for each component in the cell. Thus e.g. EHData[2] = array slice of
#      amplitudes for field component components[2], all at a single frequency.
#
#   -- EHCatalog refers to a collection (one-dimensional list) of EHData
#      entities, one for each frequency in the DFTCell---that is, a 2D matrix,
#      with rows corresponding to frequencies and columns corresponding to
#      components of EH arrays. Thus e.g. EHCatalog[3][2] = array slice of
#      amplitudes for component components[2] at frequency #3.
#
#   -- Arrays of eigenmode field amplitudes are named similarly with the
#      substitution "EH" -> "eh"
#
# Note: for now, all arrays are stored in memory. For large calculations with
# many DFT frequencies this may become impractical. TODO: disk caching.
######################################################################
class DFTCell(object):

    index=0

    ######################################################################
    ######################################################################
    ######################################################################
    def __init__(self, grid_info=None, region=None, center=origin, size=None,
                 components=None, fcen=None, df=0, nfreq=1, name=None):

        if region is not None:
            self.center, self.size, self.region = region.center, region.size, region
        elif size is not None:
            self.center, self.size, self.region = center, size, mp.Volume(center=center, size=size)
        else:
            self.center, self.size, self.region = origin, grid_info.size, mp.Volume(center=center, size=size)

        self.nHat       = region.direction if hasattr(region,'direction') else None
        self.celltype   = 'flux' if self.nHat is not None else 'fields' # TODO extend to other cases
        self.components =       components if components is not None                    \
                           else EHTransverse[self.nHat] if self.celltype is 'flux'      \
                           else Exyz
        self.fcen       = fcen
        self.df         = df if nfreq>1 else 0.0
        self.nfreq      = nfreq
        self.freqs      = [fcen] if nfreq==0 else np.linspace(fcen-0.5*df, fcen+0.5*df, nfreq)

        self.sim        = None  # mp.simulation for current simulation
        self.dft_obj    = None  # meep DFT object for current simulation

        self.EH_cache   = {}    # cache of frequency-domain field data computed in previous simulations

        self.eigencache = {}    # cache of eigenmode field data to avoid redundant recalculationsq

        DFTCell.index += 1
        self.name = name if name else '{}_{}'.format(self.celltype, DFTCell.index)

        # FIXME At present the 'xyzw' metadata cannot be computed until a mp.simulation / meep::fields
        #       object has been created, but in fact the metadata only depend on the GridInfo
        #       (resolution and extents of the computational lattice) and are independent
        #       of the specific material geometry and source configuration of any particular
        #       'fields' instance or simulation. In keeping with the spirit of 'DFTCell' it should
        #       be possible to compute the metadata once and for all right here before any mp.simulation()
        #       or meep::fields is created, but that will require some code refactoring. For the time being
        #       we punt on this until later, after a fields object has been created.
        self.xyzw = self.slice_dims = None

    ######################################################################
    # 'register' the cell with a MEEP timestepping simulation to request
    # computation of frequency-domain fields
    ######################################################################
    def register(self, sim):
        self.sim     = sim
        self.dft_obj =      sim.add_flux(self.fcen,self.df,self.nfreq,self.region) if self.celltype=='flux'   \
                       else sim.add_dft_fields(self.components, self.fcen, self.df, self.nfreq, where=self.region)

        # take the opportunity to fill in the metadata if not done yet; #FIXME to be removed as discussed above
        if self.xyzw is None:
            self.xyzw  = sim.get_dft_array_metadata(center=self.center, size=self.size)
            self.slice_dims = np.shape(self.xyzw[3])


    ######################################################################
    # Compute an array of frequency-domain field amplitudes, i.e. a
    # frequency-domain array slice, for a single field component at a
    # single frequency in the current simulation. This is like
    # mp.get_dft_array(), but 'zero-padded:' when the low-level DFT object
    # does not have data for the requested component (perhaps because it vanishes
    # identically by symmetry), this routine returns an array of the expected
    # dimensions with all zero entries, instead of a rank-0 array that prints
    # out as a single preposterously large or small floating-point number,
    # which is the not-very-user-friendly behavior of get_dft_array().
    ######################################################################
    def get_EH_slice(self, c, nf=0):
        EH = self.sim.get_dft_array(self.dft_obj, c, nf)
        return EH if np.ndim(EH)>0 else 0.0j*np.zeros(self.slice_dims)

    ######################################################################
    # Return a 1D array (list) of arrays of frequency-domain field amplitudes,
    # one for each component in this DFTCell, at a single frequency in a
    # single MEEP simulation. The simulation in question may be the present,
    # ongoing simulation (if label==None), in which case the array slices are
    # read directly from the currently active meep DFT object; or it may be a
    # previous simulation (identified by label) for which DFTCell::save_fields
    # was called at the end of timestepping.
    ######################################################################
    def get_EH_slices(self, nf=0, label=None):
        if label is None:
            return [ self.get_EH_slice(c, nf=nf) for c in self.components ]
        elif label in self.EH_cache:
            return self.EH_cache[label][nf]
        raise ValueError("data for label {} missing in get_EH_slices".format(label))

    ######################################################################
    # substract incident from total fields to yield scattered fields
    ######################################################################
    def subtract_incident_fields(self, EHT, nf=0):
        EHI = self.get_EH_slices(nf=nf, label='incident')
        for nc,c in enumerate(self.components):
            EHTData[nc] -= EHIData[nc]

    ####################################################################
    # This routine tells the DFTCell to create and save an archive of
    # the frequency-domain array slices for the present simulation---i.e.
    # to copy the frequency-domain field data out of the sim.dft_obj
    # structure and into an appropriate data buffer in the DFTCell,
    # before the sim.dft_obj data vanish when sim is deleted and replaced
    # by a new simulation. This routine should be called after timestepping
    # is complete. The given label is used to identify the stored data
    # for purposes of future retrieval.
    ######################################################################
    def save_fields(self,label):
        if label in self.EH_cache:
            raise ValueError("Data for label {} has already been saved in EH_cache",label)
        self.EH_cache[label] = [self.get_EH_slices(nf=nf) for nf in range(len(self.freqs))]

    ######################################################################
    # Return a 1D array (list) of arrays of field amplitudes for all
    # tangential E,H components at a single frequency---just like
    # get_EH_slices()---except that the sliced E and H fields are the
    # fields of eigenmode #mode.
    ######################################################################
    def get_eigenfield_slices(self, mode, nf=0):

        # look for data in cache
        tag='M{}.F{}'.format(mode,nf)
        if self.eigencache and tag in self.eigencache:
            return self.eigencache[tag]

        # data not in cache; compute eigenmode and populate slice arrays
        freq=self.freqs[nf]
        dir=self.nHat
        vol=mp.Volume(self.region.center,self.region.size)
        k_initial=mp.Vector3()
        eigenmode=self.sim.get_eigenmode(freq, dir, vol, mode, k_initial)
        eh_slices=[]
        x,y,z,w=self.xyzw[0], self.xyzw[1], self.xyzw[2], self.xyzw[3]
        xyz = [mp.Vector3(xx,yy,zz) for xx in x for yy in y for zz in z]
        for c in self.components:
            slice, nxyz = 0.0j*np.zeros(self.slice_dims), 0
            for n,ww in np.ndenumerate(w):
                slice[n]=eigenmode.amplitude(xyz[nxyz], c)
                nxyz+=1
            eh_slices.append(slice)

        # store in cache before returning
        if self.eigencache is not None:
            print("Adding eigenfields for tag {}".format(tag))
            self.eigencache[tag]=eh_slices

        return eh_slices

    ##################################################
    # compute an objective quantity, i.e. an eigenmode
    # coefficient or a scattered or total power
    ##################################################
    def EvalQuantity(self, qtype, mode, nf=0):

        w  = self.xyzw[3]
        EH = self.get_EH_slices(nf)
        if qtype.islower():
             self.subtract_incident_fields(EH,nf)

        if qtype in 'sS':
            return 0.25*np.real(np.sum(w*( np.conj(EH[0])*EH[3] - np.conj(EH[1])*EH[2]) ))
        elif qtype in 'PM':
            eh = self.get_eigenfield_slices(mode, nf)  # EHList of eigenmode fields
            eH = np.sum( w*(np.conj(eh[0])*EH[3] - np.conj(eh[1])*EH[2]) )
            hE = np.sum( w*(np.conj(eh[3])*EH[0] - np.conj(eh[2])*EH[1]) )
            sign=1.0 if qtype=='P' else -1.0
            return (eH + sign*hE)/2.0
        else: # TODO: support other types of objectives quantities?
            ValueError('unsupported quantity type')


#def flux_line(x0, y0, length, dir):
#    size=length*(xHat if dir is mp.Y else yhat/
#    return mp.FluxRegion(center=mp.Vector3(x0,y0), size=size, direction=dir)



#########################################################
# ObjectiveFunction is a simple class for evaluating
# arbitrary user-specified objective functions; its data are
# (a) a mathematical expression for the function (a
#     character string in which the names of various
#     objective quantities like 'S_0' or 'P1_1' appear), and
# (b) the dft_cells needed to compute the objective quantities.
#########################################################
class ObjectiveFunction(object):

    ######################################################################
    # try to create a sympy expression from the given string and determine
    # names for all input variables (objective quantities) needed to
    # evaluate it
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
        #/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
        self.sim=None
        #/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

    #########################################################
    # attempt to parse the name of an objective quantity like
    # 'P2_3' or 's_4'. On success, return triple
    # (qtype, eigenmode_index, region_index).
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

#########################################################
#########################################################
#########################################################
def eval_gradient(sim, dft_cells, basis, nf=0):

     gradient=0.0j*np.zeros(len(basis()))
     cell=dft_cells[-1] # design cell
     EHForward=cell.get_EH_slices(nf,label='forward')
     EHAdjoint=cell.get_EH_slices(nf) # no label->current simulation
     f=0.0j*np.ones(cell.slice_dims)
     for nc,c in enumerate(cell.components):
         if c not in Exyz:
             continue
         f+=EHForward[nc]*EHAdjoint[nc]
     return project_basis(basis,cell.xyzw,f)

#########################################################
# Given a list of DFT cells, continue timestepping until the
# frequency-domain fields have stopped changing within the
# specified relative tolerance or the time limit is reached.
#########################################################
def run_until_dfts_converged(sim, dft_cells, obj_func, basis=None):

    #last_source_time=sim.fields.last_source_time()
    last_source_time=sim.sources[0].src.swigobj.last_time_max
    max_time=adjoint_options['dft_timeout']*last_source_time()
    verbose=(adjoint_options['verbosity']=='verbose')
    reltol=adjoint_options['dft_reltol']
    check_interval=adjoint_options['dft_interval']

    # run without interruption until the sources have died out
    for cell in dft_cells:
        cell.register(sim)
    sim.init_sim()
    sim.run(until=sim.fields.last_source_time())

    # now continue running with periodic convergence checks until
    # convergence is established or we time out.
    # convergence means that all relevant quantities have stopped
    # changing to within the specified tolerances; the relevant
    # quantities are (a) values of the objective function and of
    # all objective quantities for the forward run, (b) values of
    # all gradient components for the adjoint run. The latter case
    # is distinguished from the former by a non-None value for the
    # 'basis' input parameter.
    next_check_time = sim.round_time() + check_interval
    if basis:
        Q       = eval_gradient(sim, dft_cells, basis)
        Q_names = ['b{}'.format(n) for n in range(len(basis()))]
    else:
        Q       = np.array( [obj_func.eval()] + obj_func.qvalues )
        Q_names = ['F'] + obj_func.qnames
    last_Q = Q
    while True:
        sim.run(until=next_check_time)
        next_check_time += check_interval
        next_check_time = min(next_check_time, max_time)

        if basis:
            Q = eval_gradient(sim, dft_cells, basis)
        else:
            Q = np.array( [obj_func.eval()] + obj_func.qvalues )
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
def place_adjoint_sources(sim, envelope, qname, dft_cells):

   freq     = envelope.frequency
   omega    = 2.0*np.pi*freq
   factor   = 2.0j*omega
   if callable(getattr(envelope, "fourier_transform", None)):
       factor /= envelope.fourier_transform(omega)

   nf=0
   (qtype,mode,ncell)=ObjectiveFunction.unpack_quantity_name(qname)
   cell=dft_cells[ncell]

   EH =     cell.get_EH_slices(nf=nf,label='forward')  if mode==0 \
       else cell.get_eigenfield_slices(mode=mode,nf=0)

   components = cell.components
   (x,y,z,w)  = cell.xyzw[0],cell.xyzw[1],cell.xyzw[2],cell.xyzw[3]
   shape      = [np.shape(q)[0] for q in [x,y,z]]

   if qtype in 'PM':
       sign = 1.0 if qtype=='P' else -1.0
       signs=[+1.0,-1.0,+1.0*sign,-1.0*sign]
       sim.sources+=[mp.Source(envelope, components[3-nc],
                               cell.center, cell.size,
                               amplitude=signs[nc]*factor,
                               amp_data=np.reshape(np.conj(EH[nc]),shape)
                              ) for nc in range(4)]

########################################################
# do a forward timestepping run to compute the value of
# the objective function. The return value is a tuple
# (F,Q), where F is the scalar objective-function value
# and Q is the vector of objective-quantity values.
########################################################
def get_objective(sim, obj_func, dft_cells):

    #/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
    obj_func.sim=sim
    #/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

    FQ=run_until_dfts_converged(sim, dft_cells, obj_func=obj_func)

    if adjoint_options['visualize']:
        fig=plt.figure(2)
        plt.clf()
        visualize_sim(sim)

    return FQ[0]


########################################################
# same as previous routine, but now followed by an
# adjoint run to compute the objective-function gradient
# with respect to the basis-function coefficients.
########################################################
def get_objective_and_gradient(sim, obj_func, dft_cells,
                               basis, qname=None):

    # do the forward run and save the frequency-domain fields
    # in all objective regions and in the design region
    F=get_objective(sim, obj_func, dft_cells)
    for cell in dft_cells:
        cell.save_fields('forward')

    # remove the forward sources and replace with adjoint sources
    envelope=sim.sources[0].src
    sim.reset_meep()
    sim.change_sources([])
    qname=qname if qname else obj_func.qnames[0]
    place_adjoint_sources(sim,envelope,qname,dft_cells)

    # do the adjoint run
    sim.force_complex_fields=True
    gradF =run_until_dfts_converged(sim, dft_cells, obj_func, basis=basis)

    return F, gradF
