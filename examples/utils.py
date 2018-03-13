import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import corner
from getdist import plots, MCSamples


def plot_corner(samples):
    
    ndim = samples.shape[1]    
    labels_corner =  ["$x_%s$"%i for i in range(ndim)]    
    fig = corner.corner(samples, labels=labels_corner)
    
    
def plot_getdist(samples):

    ndim = samples.shape[1]    
    names = ["x%s"%i for i in range(ndim)]
    labels =  ["x_%s"%i for i in range(ndim)]    
    
    mcsamples = MCSamples(samples=samples, 
                          names=names, labels=labels)        
    g = plots.getSubplotPlotter()
    g.triangle_plot([mcsamples], filled=True)


def eval_func_on_grid(func, xmin, xmax, ymin, ymax, nx, ny):

    # Evaluate func over grid.
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    func_eval_grid = np.zeros((nx,ny))            
    for i in range(nx):
        for j in range(ny):
            func_eval_grid[i,j] = \
                 func( np.array([x_grid[i,j], y_grid[i,j]]) )
                 
    return func_eval_grid, x_grid, y_grid
 

def plot_surface(func_eval_grid, x_grid, y_grid, samples=None, vals=None):
    # xmin, xmax, ymin, ymax, nx, ny, samples=None, ln_vals=None):
    
    # if samples is not None then ln_vals must also be not None
    # also check sizes consistent
    

    # Set up axis for surface plot.
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    
    # Create an instance of a LightSource and use it to illuminate
    # the surface.
    light = LightSource(60, 120)
    rgb = np.ones((func_eval_grid.shape[0], 
                   func_eval_grid.shape[1], 
                   3))
    illuminated_surface = \
        light.shade_rgb(rgb * np.array([0,0.0,1.0]), 
                        func_eval_grid)

    # Plot surface.
    ax.plot_surface(x_grid, y_grid, func_eval_grid, 
                    alpha=0.5, linewidth=0, antialiased=False,
                    # cmap=cm.coolwarm, 
                    facecolors=illuminated_surface)
    
    
    # Plot contour.
    # cset = ax.contour(x_grid, y_grid, func_eval_grid, 
    #                   zdir='z', offset=-0.5, cmap=cm.coolwarm)  
                
                
                
                
    xmin = np.min(x_grid)
    xmax = np.max(x_grid) 
    ymin = np.min(y_grid)
    ymax = np.max(y_grid)
                
                
                  
    # # Plot samples.
    # i_chain = 0
    if samples is not None:
        xplot = samples[:,0]
        yplot = samples[:,1]
        # Manually remove samples outside of plot region 
        # (since Matplotlib clipping cannot do this in 3D; see 
        # https://github.com/matplotlib/matplotlib/issues/749).
        xplot[xplot < xmin] = np.nan
        xplot[xplot > xmax] = np.nan        
        yplot[yplot < ymin] = np.nan
        yplot[yplot > ymax] = np.nan        
        zplot = vals
        ax.scatter(xplot, yplot, zplot, c='r', s=5, marker='.')
    # 
    # Define additional plot settings.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_zlim(-20.0, 1.0)
    ax.view_init(elev=15.0, azim=110.0)        
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    # ax.set_zlabel('$ln L$')
    return ax
    
    
    
def plot_image(func_eval_grid, x_grid, y_grid, samples=None):

    plt.figure()
    ax = plt.imshow(func_eval_grid, origin='lower', 
               extent=[np.min(x_grid), np.max(x_grid), 
                       np.min(y_grid), np.max(y_grid)])
               # vmin=-100.0, vmax=0.0)
    # plt.contour(x_grid, y_grid, func_eval_grid, cmap=cm.coolwarm)
    
    if samples is not None:
        
        plt.plot(samples[:,0], 
                 samples[:,1], 
                 'r.', markersize=1)
                 
    plt.colorbar()
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')   
    
    return ax