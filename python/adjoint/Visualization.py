###################################################
# visualize.py -- various routines for visualizing
# the inputs and outputs of pymeep calculations
###################################################
import warnings
import numpy as np
import theano
import sympy
from sympy.printing.theanocode import theano_function

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import cm
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import axes3d

import meep as mp

########################################################
# this routine configures some global matplotlib settings
# (as distinct from the plot-specific settings handled
# by the plot_options dicts). if you intend to set your
# own values of these parameters, pass set_rmecParams=False
# to visualize_sim.
########################################################
def set_meep_rcParams():
    plt.rc('xtick')
    plt.rc('ytick')
    plt.rc('font',  size=40)
    plt.rc('text', usetex=True)

########################################################
# plot_options(type) returns a dict of user=customizable
# plotting options initialized appropriately for
# given type of data
########################################################
def_plot_options={ 'line_color'     : [1.0,0.0,1.0],
                   'line_width'     : 4.0,
                   'line_style'     : '-',
                   'boundary_width' : 2.0,
                   'boundary_color' : [0.0,1.0,0.0],
                   'boundary_style' : '--',
                   'z_offset'       : 0.0,
                   'zmin'           : 0.60,
                   'zmax'           : 1.00,
                   'fill_color'     : 'none',
                   'alpha'          : 1.0,
                   'cmap'           : cm.plasma,
                   'fontsize'       : 40,
                   'colorbar_shrink': 0.60,
                   'colorbar_pad'   : 0.04,
                  }

def plot_options(type='dft', plot3D=False):
    options=dict(def_plot_options)
    if type=='eps':
        options['cmap']=cm.Blues
        options['line_width']=0.00
        options['plot_method']='contourf' # or 'pcolormesh' or 'imshow'
        options['num_contours']=100
        options['colorbar_shrink']:0.75
    elif type=='src':
        options['line_color']=[0.0,1.0,1.0]
        options['line_width']=4.0
        options['zmin']=options['zmax']=0.0
    elif type=='pml':
        options['boundary_color']='none'
        options['boundary_width']=0.0
        options['fill_color']=(0.25 if plot3D else 0.75)*np.ones(3)
        options['alpha']=0.25
    elif type=='flux':
        options['line_color']=[1.0,0.0,1.0]
        options['line_width']=4.0
        options['boundary_color']=[0.0,0.0,0.0]
        options['boundary_width']=2.0
        options['boundary_style']='--'
    elif type=='fields':
        options['cmap']=cm.plasma
        options['line_width']=0.0
        options['alpha']=0.5
        options['z_offset']=0.5
        options['num_contours']=100
    return options

###################################################
###################################################
###################################################
def get_text_size(fig, label, fontsize):
  r = fig.canvas.get_renderer()
  t = plt.text(0.5, 0.5, label, fontsize=fontsize)
  bb = t.get_window_extent(renderer=r)
  return bb.width, bb.height

###################################################
# visualize epsilon distribution. eps_min, eps_max
# are optional upper and lower clipping bounds.
# If we are in 3D, the permittivity is plotted using
# plot_surface.
# Otherwise, the permittivity is plotted using
# imshow (if use_imshow==True) or using pcolormesh
# (by default).
###################################################
def plot_eps(sim, eps_min=None, eps_max=None, options=None, plot3D=False):

    options       = options if options else plot_options('eps')
    cmap          = options['cmap']
    edgecolors    = options['line_color']
    linewidth     = options['line_width']
    alpha         = options['alpha']
    fontsize      = options['fontsize']
    plot_method   = options['plot_method']
    num_contours  = options['num_contours']
    shading       = 'gouraud' if linewidth==0.0 else 'none'
    interpolation = 'gaussian' if linewidth==0.0 else 'none'

    eps=np.transpose(sim.get_epsilon())
    if eps_min is not None or eps_max is not None:
        eps=np.clip(eps,eps_min,eps_max)
    eps_min, eps_max=np.min(eps), np.max(eps)
    (x,y,z,w)=sim.get_array_metadata()
    extent=(min(x), max(x), min(y), max(y))

    fig=plt.gcf()
    ax = fig.gca(projection='3d') if plot3D else fig.gca()
    cb = None
    if plot3D:  # check if 3D plot
        X, Y = np.meshgrid(x, y)
        zmin = 0.0
        zmax = max(sim.cell_size.x, sim.cell_size.y)
        Z0   = zmin + options['z_offset']*(zmax-zmin)
        img  = ax.contourf(X, Y, eps, num_contours, zdir='z', offset=Z0,
                           vmin=eps_min, vmax=eps_max, cmap=cmap, alpha=alpha)
        ax.set_zlim3d(zmin, zmax)
        ax.set_zticks([])
        pad=options['colorbar_pad']
        shrink=options['colorbar_shrink']
        cb=fig.colorbar(img, shrink=shrink, pad=pad)
    elif plot_method=='imshow':
        img = plt.imshow(np.transpose(eps), extent=extent, cmap=cmap,
                         interpolation=interpolation, alpha=alpha)
    elif plot_method=='pcolormesh':
        img = plt.pcolormesh(x,y,np.transpose(eps), cmap=cmap, shading=shading,
                             edgecolors=edgecolors, linewidth=linewidth, alpha=alpha)
    else:
        X, Y = np.meshgrid(x, y)
        img  = ax.contourf(X, Y, eps, num_contours, vmin=eps_min, vmax=eps_max,
                           cmap=cmap, alpha=alpha)

    ax.set_xlabel(r'$x$', fontsize=fontsize, labelpad=fontsize)
    ax.set_ylabel(r'$y$', fontsize=fontsize, labelpad=fontsize, rotation=0)
    ax.tick_params(axis='both', labelsize=0.75*fontsize)
    cb=cb if cb else fig.colorbar(img)
    cb.set_label(r'$\epsilon$',fontsize=1.5*fontsize,rotation=0,labelpad=0.5*fontsize)
    cb.ax.tick_params(labelsize=0.75*fontsize)

##################################################
# plot_volume() adds a polygon representing a given
# mp.Volume to the current 2D or 3D plot.
##################################################
def plot_volume(sim, vol=None, center=None, size=None,
                options=None, plot3D=False, label=None):

    options = options if options else plot_options()

    fig=plt.gcf()
    ax=fig.gca(projection='3d') if plot3D else fig.gca()

    if vol:
       center, size = vol.center, vol.size
    v0=np.array([center.x, center.y])
    dx,dy=np.array([0.5*size.x,0.0]), np.array([0.0,0.5*size.y])
    if plot3D:
        zmin,zmax = ax.get_zlim3d()
        z0 = zmin + options['z_offset']*(zmax-zmin)

    ##################################################
    # add polygon(s) to the plot to represent the volume
    ##################################################
    def add_to_plot(c):
        ax.add_collection3d(c,zs=z0,zdir='z') if plot3D else ax.add_collection(c)

    if size.x==0.0 or size.y==0.0:    # zero thickness, plot as line
        polygon = [ v0+dx+dy, v0-dx-dy ]
        add_to_plot( LineCollection( [polygon], colors=options['line_color'],
                                     linewidths=options['line_width'],
                                     linestyles=options['line_style']
                                   )
                   )
    else:
        if options['fill_color'] is not 'none': # first copy: faces, no edges
            polygon = np.array([v0+dx+dy, v0-dx+dy, v0-dx-dy, v0+dx-dy])
            pc=PolyCollection( [polygon], linewidths=0.0)
            pc.set_color(options['fill_color'])
            pc.set_alpha(options['alpha'])
            add_to_plot(pc)
        if options['boundary_width']>0.0: # second copy: edges, no faces
            closed_polygon = np.array([v0+dx+dy, v0-dx+dy, v0-dx-dy, v0+dx-dy, v0+dx+dy])
            lc=LineCollection([closed_polygon])
            lc.set_linestyle(options['boundary_style'])
            lc.set_linewidth(options['boundary_width'])
            lc.set_edgecolor(options['boundary_color'])
            add_to_plot(lc)

    ##################################################
    ##################################################
    ##################################################
    if label:
        if plot3D:
            ax.text(center.x, center.y, z0, label,
                    fontsize=options['fontsize'], color=options['line_color'])
        else:
            ax.text(center.x, center.y, label,
                    fontsize=options['fontsize'], color=options['line_color'])

##################################################
# Plot one or more curves,
##################################################
def plot_data_curves(sim,center=None,size=None,superpose=True,
                     data=None, labels=None, options=None,
                     dmin=None, dmax=None):

    if size.x>0 and size.y>0:
        msg="plot_data_curves: expected zero-width region, got {}x{} (skipping)"
        warnings.warn(msg.format(size.x,size.y),RuntimeWarning)
        return
    if np.ndim(data[0])!=1:
        msg="plot_data_curves: expected 1D data arrays, got {} (skipping)"
        warnings.warn(msg.format(np.shape(data[0])),RuntimeWarning)
        return

    options=options if options else plot_options('flux')
    lw=options['line_width']
    lc=options['line_color']
    zeta_min=options['zmin']
    zeta_max=options['zmax']
    draw_baseline=(options['boundary_width']>0.0)

    kwargs=dict()
    if 'line_color' in options:
       kwargs['color']=options['line_color']
    if 'line_width' in options:
       kwargs['linewidth']=options['line_width']
    if 'line_style' in options:
       kwargs['linestyle']=options['line_style']

    # construct horizontal axis
    ii=1 if size.x==0 else 0
    hstart,hend = (center-0.5*size).__array__()[0:2], (center+0.5*size).__array__()[0:2]
    hmin,hmax=hstart[ii],hend[ii]
    haxis = np.linspace(hmin, hmax, len(data[0]))

    # if we are superposing the curves onto a simulation-geometry
    # visualization plot, construct the appropriate mapping that
    # squeezes the full vertical extent of the curve into the
    # z-axis interval [zmin, zmax]
    if superpose:
        ax=plt.gcf().gca(projection='3d')
        (zfloor,zceil)=ax.get_zlim()
        zmin=zfloor + zeta_min*(zceil-zfloor)
        zmax=zfloor + zeta_max*(zceil-zfloor)
        z0,dz=0.5*(zmax+zmin),(zmax-zmin)
        dmin=dmin if dmin else np.min(data)
        dmax=dmax if dmax else np.max(data)
        d0,dd=0.5*(dmax+dmin),(dmax-dmin)
        zs = center[1-ii]
        zdir='x' if size.x==0 else 'y'
        if draw_baseline:
            lc=LineCollection( [[hstart,hend]], colors=options['boundary_color'],
                                linewidths=options['boundary_width'],
                                linestyles=options['boundary_style']
                             )
            ax.add_collection3d(lc,zs=z0,zdir='z')

    for n in range(len(data)):
        kwargs['label']=None if not labels else labels[n]
        if superpose:
            ax.plot(haxis,z0+(data[n]-d0)*dz/dd, zs=zs, zdir=zdir, **kwargs)
        else:
            plt.plot(haxis,data[n],**kwargs)

##################################################
##################################################
##################################################
def visualize_source_distribution(sim, standalone=False, options=None):
    if not mp.am_master():
        return

    options=options if options else plot_options('src')
    for ns,s in enumerate(sim.sources):
        sc,ss=s.center,s.size
        J2=sum([abs2(sim.get_source_slice(c,center=sc,size=ss)) for c in Exyz])
#       M2=sum([abs2(sim.get_source_slice(c,center=sc,size=ss)) for c in Hxyz])
        if standalone:
            if ns==0:
                plt.figure()
                plt.title('Source regions')
            fig.subplot(len(sim.sources),1,ns+1)
            fig.title('Currents in source region {}'.format(ns))
#        plot_data_curves(sim,superpose,[J2,M2],labels=['||J||','||M||'],
#                         styles=['bo-','rs-'],center=sc,size=ssu
        plot_data_curves(sim,center=sc,size=ss, superpose=not standalone,
                         data=[J2], labels=['J'], options=options)

##################################################
##################################################
##################################################
def visualize_dft_flux(sim, standalone=False, flux_cells=[],
                       options=None, nf=0):
    if not mp.am_master():
        return
    options=options if options else plot_options('flux')
    if len(flux_cells)==0:
        for cell in sim.dft_objects:
            if is_flux_cell(cell):
               flux_cells+=[cell]
    maxflux=0.0
    S_list=[]
    for n, cell in enumerate(flux_cells):    # first pass to compute flux data
        cn,sz=mp.get_center_and_size(cell.where)
        (x,y,z,w,c,EH)=unpack_dft_cell(sim,cell,nf=nf)
        S=0.25*np.real(w*(np.conj(EH[0])*EH[3] - np.conj(EH[1])*EH[2]))
        maxflux=max(maxflux, np.max(abs(S)))
        S_list.append(S)

    for n, cell in enumerate(flux_cells):    # second pass to plot
        if standalone:
            if n==0:
                plt.figure()
                plt.title('Poynting flux')
            plt.subplot(length(flux_cells),1,n)
            plt.gca().set_title('Flux cell {}'.format(n))
        cn,sz=mp.get_center_and_size(cell.where)
        plot_data_curves(sim, center=cn, size=sz, data=[S_list[n]],
                         superpose=not standalone, options=options,
                         labels=['flux through cell {}'.format(n)],
                         dmin=-maxflux,dmax=maxflux)
    plt.show(False)

##################################################
##################################################
##################################################
def fc_name(c,which):
    name=mp.component_name(c)
    return name if which=='scattered' else str(name[0].upper())+str(name[1])

def field_func_array(fexpr,x,y,z,w,cEH,EH):
    if fexpr=='re(Ex)':
        return np.real(EH[0])
    if fexpr=='im(Ex)':
        return np.imag(EH[0])
    if fexpr=='re(Ey)':
        return np.real(EH[1])
    if fexpr=='im(Ey)':
        return np.imag(EH[1])
    if fexpr=='re(Ez)':
        return np.real(EH[2])
    if fexpr=='im(Ez)':
        return np.imag(EH[2])
    if fexpr=='re(Hx)':
        return np.real(EH[3])
    if fexpr=='im(Hx)':
        return np.imag(EH[3])
    if fexpr=='re(Hy)':
        return np.real(EH[4])
    if fexpr=='im(Hy)':
        return np.imag(EH[4])
    if fexpr=='re(Hz)':
        return np.real(EH[5])
    if fexpr=='im(Hz)':
        return np.imag(EH[5])
    if fexpr=='abs2(H)':
        return abs2(EH[3]) + abs2(EH[4]) + abs2(EH[5])
    if True: # fexpr=='abs2(E)':
        return abs2(EH[0]) + abs2(EH[1]) + abs2(EH[2])

##################################################
##################################################
##################################################
Eabs2_func='abs2(E)'
def visualize_dft_fields(sim, standalone=False, field_cells=[],
                         field_func_str=Eabs2_func, options=None, nf=0):
    if not mp.am_master():
        return

    if len(field_cells)==0:
        for cell in sim.dft_objects:
            if dft_cell_type(cell)=='fields':
               field_cells+=[cell]
    if len(field_cells)==0:
        return

    if not standalone and not isinstance(plt.gcf().gca(),axes3d.Axes3D):
        standalone=True
        warnings.warn("current plot axes are not 3D; setting standalone=True in visualize_dft_fields")

#    try:
#        (Ex,Ey,Ez,Hx,Hy,Hz,x,y,z,w) = sympy.symbols('Ex,Ey,Ez,Hx,Hy,Hz,x,y,z,w')
#        f_expr = sympy.sympify(f_str)
#        dim=len(np.shape(Ex))
#        d64, c128='float64', 'complex128'
#        f_evaluator = theano_function([Ex,Ey,Ez,Hx,Hy,Hz,x,y,z,w], f_expr,
#                                      dims={Ex:dim, Ey:dim, Ez:dim,
#                                            Hx:dim, Hy:dim, Hz:dim,
#                                             x:dim,  y:dim,  z:dim, w:dim},
#                                      dtypes={Ex: c128, Ey: c128, Ez: c128,
#                                              Hx: c128, Hy: c128, Hz: c128,
#                                               x: d64, y:d64, z:d64, w:d64}
#                                     )
#    except:
#        raise ValueError("failed to parse field function {}".format(field_func))

    options       = options if options else plot_options('fields')
    cmap          = options['cmap']
    alpha         = options['alpha']
    num_contours  = options['num_contours']
    fontsize      = options['fontsize']

    for n, cell in enumerate(field_cells):
        (x,y,z,w,cEH,EH)=unpack_dft_cell(sim,cell,nf=nf)
        data=field_func_array(field_func_str,x,y,z,w,cEH,EH)
        X, Y = np.meshgrid(x, y)
        if standalone:
            if n==0:
                plt.figure()
                plt.suptitle('DFT fields (function={})'.format(field_func_str))
            plt.subplot(len(field_cells),1,n)
            ax=plt.gca()
            ax.set_title('Field cell {}'.format(n))
            ax.set_xlabel(r'$x$', fontsize=fontsize, labelpad=fontsize)
            ax.set_ylabel(r'$y$', fontsize=fontsize, labelpad=fontsize, rotation=0)
            ax.tick_params(axis='both', labelsize=0.75*fontsize)
            img = ax.contourf(X,Y,np.transpose(data),num_contours,
                              cmap=cmap,alpha=alpha)
            cb=plt.colorbar(img)
            #cb.set_label(r'$\epsilon$',fontsize=1.5*fontsize,rotation=0,labelpad=0.5*fontsize)
            #cb.ax.tick_params(labelsize=0.75*fontsize)
        else:
            fig = plt.gcf()
            ax  = fig.gca(projection='3d')
            (zmin,zmax)=ax.get_zlim()
            Z0=zmin+options['z_offset']*(zmax-zmin)
            img = ax.contourf(X, Y, np.transpose(data), num_contours,
                              cmap=cmap, alpha=alpha, zdir='z', offset=Z0)
            pad=options['colorbar_pad']
            shrink=options['colorbar_shrink']
            cb=fig.colorbar(img, shrink=shrink, pad=pad, orientation='horizontal')
            cb.set_label(r'$|\mathbf{E}|^2$',fontsize=1.0*fontsize,rotation=0,labelpad=0.5*fontsize)
            cb.ax.tick_params(labelsize=0.75*fontsize)
    plt.show(False)

##################################################
##################################################
##################################################
def visualize_sim(sim, fig=None, plot3D=None,
                  eps_min=0.0, eps_max=None,
                  eps_options=None, src_options=None,
                  pml_options=None, dft_options=None,
                  flux_options=None, field_options=None,
                  set_rcParams=True, plot_dft_data=None):

    if not mp.am_master():
        return

    # if plot3D not specified, set it automatically: false
    # if we are plotting only the geometry (at the beginning
    # of a simulation), true if we are also plotting results
    # (at the end of a simulation).
    sources_finished = sim.round_time() > sim.fields.last_source_time()
    if plot3D is None:
        plot3D=sources_finished

    ######################################################
    # create figure and set some global parameters, unless
    #  the caller asked us not to
    ######################################################
    if fig is None:
        fig=plt.gcf()
        fig.clf()
        if set_rcParams:
            set_meep_rcParams()
    ax = axes3d.Axes3D(fig) if plot3D else fig.gca()
    if not plot3D:
        ax.set_aspect('equal')
        plt.tight_layout()

    ##################################################
    # plot permittivity
    ##################################################
    eps_options = eps_options if eps_options else plot_options(type='eps', plot3D=plot3D)
    plot_eps(sim, eps_min=eps_min, eps_max=eps_max, options=eps_options, plot3D=plot3D)

    ###################################################
    ## plot source regions and optionally source amplitudes
    ###################################################
    src_options = src_options if src_options else plot_options(type='src', plot3D=plot3D)
    for s in sim.sources:
        plot_volume(sim, center=s.center, size=s.size,
                    options=src_options, plot3D=plot3D)
    if src_options['zmin']!=src_options['zmax']:
        visualize_source_distribution(sim, standalone=not plot3D, options=src_options)

    ###################################################
    ## plot PML regions
    ###################################################
    pml_options = pml_options if pml_options else plot_options(type='pml', plot3D=plot3D)
    if sim.boundary_layers and hasattr(sim.boundary_layers[0],'thickness'):
        dpml    = sim.boundary_layers[0].thickness
        sx, sy  = sim.cell_size.x, sim.cell_size.y
        y0, x0  = mp.Vector3(0.0, 0.5*(sy-dpml)), mp.Vector3(0.5*(sx-dpml), 0.0)
        ns, ew  = mp.Vector3(sx-2*dpml, dpml),    mp.Vector3(dpml,sy)
        centers = [ y0, -1*y0, x0, -1*x0 ]   # north, south, east, west
        sizes   = [ ns,    ns, ew,    ew ]
        for c,s in zip(centers,sizes):
            plot_volume(sim, center=c, size=s,
                        options=pml_options, plot3D=plot3D)

    ###################################################
    ## plot DFT cells, with index labels for flux cells
    ###################################################
    dft_options = dft_options if dft_options else plot_options(type='dft',plot3D=plot3D)
    for nc, c in enumerate(sim.dft_objects):
        plot_volume(sim,center=c.regions[0].center,size=c.regions[0].size,
                    options=dft_options, plot3D=plot3D,
                    label=(str(nc) if dft_cell_type(c)=='flux' else None))

    ###################################################
    ###################################################
    ###################################################
    if plot_dft_data is None:
        plot_dft_data=sources_finished

    if plot_dft_data:
        visualize_dft_flux(sim, options=flux_options)
        visualize_dft_fields(sim, options=field_options)

    plt.show(False)
    return fig
