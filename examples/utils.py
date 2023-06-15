import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import corner
from getdist import plots, MCSamples
import getdist

def plot_corner(samples, labels=None):
    """
    Plot triangle plot of marginalised distributions using corner package.

    Args:
        - samples: 
            2D array of shape (ndim, nsamples) containing samples.
        - labels: 
            Array of strings containing axis labels.
    Returns:
        - None
    """

    ndim = samples.shape[1]
    if labels is None:
        labels_corner =  ["$x_%s$"%i for i in range(ndim)]
    else:
        labels_corner = labels
    fig = corner.corner(samples, labels=labels_corner)


def plot_getdist(samples, labels=None):
    """
    Plot triangle plot of marginalised distributions using getdist package.

    Args:
        - samples: 
            2D array of shape (ndim, nsamples) containing samples.
        - labels: 
            Array of strings containing axis labels.

    Returns:
        - None
    """

    getdist.chains.print_load_details = False

    ndim = samples.shape[1]
    names = ["x%s"%i for i in range(ndim)]
    if labels is None:
        labels =  ["x_%s"%i for i in range(ndim)]

    mcsamples = MCSamples(samples=samples,
                          names=names, labels=labels)
    g = plots.getSubplotPlotter()
    g.triangle_plot([mcsamples], filled=True)

def plot_getdist_compare(samples1, samples2, labels=None):
    """
    Plot triangle plot of marginalised distributions using getdist package.

    Args:
        - samples: 
            2D array of shape (ndim, nsamples) containing samples.
        - labels: 
            Array of strings containing axis labels.

    Returns:
        - None
    """

    getdist.chains.print_load_details = False

    ndim = samples1.shape[1]
    names = ["x%s"%i for i in range(ndim)]
    if labels is None:
        labels =  ["x_%s"%i for i in range(ndim)]

    mcsamples1 = MCSamples(samples=samples1,
                          names=names, labels=labels, label='Posterior samples')
    
    mcsamples2 = MCSamples(samples=samples2,
                          names=names, labels=labels, label='Compressed NVP samples')
    
    g = plots.getSubplotPlotter()
    g.triangle_plot([mcsamples1, mcsamples2], filled=True)


def eval_func_on_grid(func, xmin, xmax, ymin, ymax, nx, ny):
    """
    Evalute 2D function on a grid.

    Args:
        - func: 
            Function to evalate.
        - xmin: 
            Minimum x value to consider in grid domain.
        - xmax: 
            Maximum x value to consider in grid domain.
        - ymin: 
            Minimum y value to consider in grid domain.
        - ymax: 
            Maximum y value to consider in grid domain.
        - nx: 
            Number of samples to include in grid in x direction.
        - ny: 
            Number of samples to include in grid in y direction.

    Returns:
        - func_eval_grid: 
            Function values evaluated on the 2D grid.
        - x_grid: 
            x values over the 2D grid.
        - y_grid: 
            y values over the 2D grid.
    """

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


def plot_surface(func_eval_grid, x_grid, y_grid, samples=None, vals=None,
                 contour_z_offset=None, contours=None, alpha=0.3):
    """
    Plot surface defined by 2D function on a grid.  

    Samples may also be optionally plotted.

    Args:
        - func_eval_grid: 
            Function evalated over 2D grid.
        - x_grid: 
            x values over the 2D grid.
        - y_grid: 
            y values over the 2D grid.
        - samples: 
            2D array of shape (ndim, nsamples) containing samples.
        - vals: 
            1D array of function values at sample locations.  Both samples and 
            vals must be provided if they are to be plotted.
        - contour_z_offset: 
            If not None then plot contour in plane specified by z offset.
        - contours: 
            Values at which to draw contours (must be in increasing order).
        - alpha:
            Opacity of surface plot.

    Returns:
        - ax: 
            Plot axis.
    """

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
                    alpha=alpha, linewidth=0, antialiased=False,
                    # cmap=cm.coolwarm,
                    facecolors=illuminated_surface)

    # Plot contour.
    if contour_z_offset is not None:
        if contours is not None:
            cset = ax.contour(x_grid, y_grid, func_eval_grid, contours,
                              zdir='z', offset=contour_z_offset,
                              cmap=cm.coolwarm)
        else:
            cset = ax.contour(x_grid, y_grid, func_eval_grid,
                              zdir='z', offset=contour_z_offset,
                              cmap=cm.coolwarm)

    # Set domain.
    xmin = np.min(x_grid)
    xmax = np.max(x_grid)
    ymin = np.min(y_grid)
    ymax = np.max(y_grid)


    # # Plot samples.
    if samples is not None and vals is not None:
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

    # Define additional plot settings.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.view_init(elev=15.0, azim=110.0)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlim(zmin=contour_z_offset)

    return ax


def plot_image(func_eval_grid, x_grid, y_grid, samples=None,
               colorbar_label=None, plot_contour=False, 
               contours=None, markersize=1.0):
    """
    Plot image defined by 2D function on a grid.  

    Samples may also be optionally plotted.

    Args:
        - func_eval_grid: 
            Function evalated over 2D grid.
        - x_grid: 
            x values over the 2D grid.
        - y_grid: 
            y values over the 2D grid.
        - samples: 
            2D array of shape (ndim, nsamples) containing samples.
        - colorbar_label: 
            Text label to include on colorbar.
        - contours: 
            Values at which to draw contours (must be in increasing order).
        - markersize:
            Size of markers for plotting overlaid samples.

    Returns:
        - ax: 
            Plot axis.
    """

    plt.figure()
    ax = plt.imshow(func_eval_grid, origin='lower', aspect='auto',
                    extent=[np.min(x_grid), np.max(x_grid),
                            np.min(y_grid), np.max(y_grid)])

    if plot_contour:
        if contours is not None:
            plt.contour(x_grid, y_grid, func_eval_grid, contours,
                        cmap=cm.coolwarm)
        else:
            plt.contour(x_grid, y_grid, func_eval_grid, cmap=cm.coolwarm)

    if samples is not None:
        plt.plot(samples[:,0],
                 samples[:,1],
                 'r.', markersize=markersize)

    if colorbar_label is not None:
        plt.colorbar(label=colorbar_label)
    else:
        plt.colorbar()

    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')

    return ax


def plot_realisations(mc_estimates, std_estimated,
                      analytic_val=None, analytic_text=None):
    """
    Violin plot of estimated quantity from Monte Carlo (MC) simulations, 
    compared with error bar from estimated standard deviation.
    
    Also plot analytic value if specified.

    Args:
        - mc_estimates: 
            1D array of quanties estimate many times by MC simulation.
        - std_estimate: 
            Standard deviation estimate to be compared with standard deviation 
            from MC simulations.
        - analytic_val: 
            Plot horizonal line if analytic value of quantity estimated is 
            provided.
        - analytic_text: 
            Text to include next to line specifying analytic value, if provided.

    Returns:
        - ax: 
            Plot axis.
    """

    mean = np.mean(mc_estimates)
    std_measured = np.std(mc_estimates)

    plot_aspect_ratio = 1.33
    plot_x_size = 9

    fig, ax = plt.subplots(figsize=(plot_x_size,
                                    plot_x_size/plot_aspect_ratio))

    ax.violinplot(mc_estimates, showmeans=False, showmedians=False,
            showextrema=True, bw_method=1.0)

    if analytic_val is not None:
        plt.plot(np.arange(4),np.zeros(4)+analytic_val, 'r--')
        ymin, ymax = ax.get_ylim()
        ax.text(1.8, analytic_val+(ymax-ymin)*0.03, analytic_text, color='red')

    plt.errorbar(np.zeros(1)+1.0, mean, yerr=std_measured,
        fmt='--o', color='C4', capsize=7, capthick=3,
        linewidth=3, elinewidth=3)
    plt.errorbar(np.zeros(1)+1.5, mean, yerr=std_estimated,
        fmt='--o', color='C2', capsize=7, capthick=3,
        linewidth=3, elinewidth=3)

    ymin, ymax = ax.get_ylim()
    print("ymim = {}, ymax = {}".format(ymin, ymax))
    if ymin < 0:
        ymin = 0
    ax.set_ylim([ymin, ymax])

    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks([1.0, 1.5])
    ax.set_xticklabels(['Measured', 'Estimated'])

    ax.set_xlim([0.5, 2.0])

    return ax
