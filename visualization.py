import numpy as np
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText


def add_box(text, ax, fontsize = 8, loc = 2):
    """
    Add box to a panel of a figure (e.g., containing "(a)", "(b)", ... in papers).
    Default location top left.
    """
    box = AnchoredText(text, prop = dict(size = fontsize), frameon = True, loc = loc)
    box.patch.set_boxstyle("round, pad = 0., rounding_size = 0.2")
    ax.add_artist(box)
    

def forceSquare(ax):
    """
    Force an axis to be square
    """
    try:
        imag = ax.get_images()
        extent = imag[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
    except IndexError:
        # won't work for an empty panel
        print("Couldn't force-square an empty axis")
        

def map_imshow(map_2d, projection = ccrs.Robinson(25),
               xmin = -9.375, xmax = 36.875, ymin = 29.875, ymax = 71.875,
               cmap = "Spectral_r", vmin = None, vmax = None, norm = None,
               cbar = {"show": True}, borders = False, water_hatch = '////////',
               line_width = 0.8, line_color = "black", lake_color = "none", fill_color = "white",
               grid_width = 0.3, grid_delta = 10, grid_labels = [], grid_label_size = 8,
               resol = "10m", return_object = False, title = None, fontsize = 10,
               figsize = (3,3), dpi = 150, ax = None, filename = None):
    
    """
    Plot a map of a 2d xarray
    
    :param xarray map_2d: Map to plot (e.g., soil moisture, LAI, ...) with lat-lon information.
    :param ccrs.proj projection: Projection of the map. Defaults to a good value for Europe.
                                 Use ccrs.Orthographic(-10, 45) to plot a globe.
    :param int xmin/xmax/ymin/ymax: Corners of the domain. Defaults to Europe.
    :param string cmap: Colormap for the plot.
    :param float vmin/vmax: Range for the colormap.
    :param matplotlib.color.norm norm: Norm for the colormap.
    :param dict cbar: Settings for the colorbar. Must contain key "show". Optional keys are "orientation", 
                      "extend", "label", "title", "ticks", "tick_labels", "minorticks".
    :param bool borders: Whether to plot borders.
    :param string water_hatch: Hatch pattern for the water (None for no hatch pattern).
    :param float line_width: Linewidth used for e.g. coastlines, edges of lakes, ...
    :param string line_color: Color of e.g. coastlines.
    :param string fill_color: Color of water and missing values.
    :param grid_width: Linewidth of the grid. Choose 0 for no grid.
    :param float grid_delta: Distance between grid lines (in degrees).
    :param bool grid_labels: List with positions (e.g., ["bottom", "left"]) for lat/lon labels (empty list for no labels).
    :param float grid_label_size: Fontsize of the grid labels, if any.
    :param str resol: Resolution of the coastlines. One of '10m', '50m', or '110m'.
    :param return_object: Whether to return the plotted object (can be useful when plotting one
                          colorbar for several panels, for example).
    :param string title: Title of the axis.
    :param float fontsize: Size of the font for the title.
    :param tuple figsize: Dimensions of the figure (if ax is None).
    :param float dpi: Dpi of the figure (if ax is None).
    :param matplotlib.ax ax: Axis on which to plot the map. Choose None to create new axis.
    :param string filename: Filename of the figure. Choose None if it shouldn't be saved.
    """
    
    # if no existing axis is provided: create a new figure
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = plt.subplot(111, projection = projection)
        fig.patch.set_alpha(0)
        ax.patch.set_color("white")
        ax.patch.set_alpha(1)
        
    # plot the map
    cs = ax.pcolormesh(map_2d.lon, map_2d.lat, map_2d.data, cmap = cmap,
                       transform = ccrs.PlateCarree(), vmin = vmin, vmax = vmax,
                       norm = norm, rasterized = True)
    cs.set_rasterized(True) # necessary for good pdf rendering
    
    # add coastlines
    #ax.coastlines(resolution = resol, linewidth = line_width, rasterized = True)
    
    # add lakes and oceans
    lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', scale = resol, 
                                                edgecolor = line_color, linewidth = line_width,
                                                facecolor = lake_color)
    ax.add_feature(lakes, rasterized = True)
    ax.add_feature(cf.OCEAN.with_scale(resol), facecolor = fill_color,
                   hatch = water_hatch, edgecolor = 'white', zorder = 0, rasterized = True)
    # ax.add_feature(cf.LAKES.with_scale(resol), facecolor = fill_color,
    #                hatch = water_hatch, edgecolor = 'black', zorder = 0, rasterized = True)
    ax.add_feature(cf.COASTLINE, linewidth = line_width, rasterized = True)

    
    # draw borders
    if borders:
        ax.add_feature(cf.BORDERS, linewidth = line_width)#, linestyle = "dashdot")
        ax.add_feature(cf.STATES, linewidth = line_width/2)

    
    # add grid
    gl = ax.gridlines(linewidth = grid_width, color = 'black', draw_labels = False if grid_labels == [] else True, alpha = 0.25)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, grid_delta))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, grid_delta))
    gl.bottom_labels = True if "bottom" in grid_labels else False
    gl.top_labels = True if "top" in grid_labels else False
    gl.left_labels = True if "left" in grid_labels else False
    gl.right_labels = True if "right" in grid_labels else False
    gl.xlabel_style = {'size': grid_label_size}
    gl.ylabel_style = {'size': grid_label_size}
    
    # set the extent of the map
    if (xmin is not None) and (xmax is not None) and (ymin is not None) and (ymax is not None):
        ax.set_extent([xmin, xmax, ymin, ymax])
    
    # add colorbar
    if cbar["show"]:
        divider = make_axes_locatable(ax)
        
        # settings      
        corientation = cbar["orientation"] if "orientation" in cbar else "vertical"
        cextend = cbar["extend"] if "extend" in cbar else "neither"
        clabel = cbar["label"] if "label" in cbar else None
        ctitle = cbar["title"] if "title" in cbar else None
        cticks = cbar["ticks"] if "ticks" in cbar else None
        ctick_draw = cbar["tick_draw"] if "tick_draw" in cbar else True
        ctick_labels = cbar["tick_labels"] if "tick_labels" in cbar else None
        cminorticks = cbar["minorticks"] if "minorticks" in cbar else True
        
        cax = divider.append_axes("right" if corientation == 'vertical' else "bottom", size = "5%", 
                                  pad = 0.05, axes_class = plt.Axes)
        cbar = plt.colorbar(cs, orientation = corientation, extend = cextend, cax = cax)
        cbar.solids.set_rasterized(True) # for good pdf rendering
        cbar.ax.set_ylabel(clabel) if corientation == 'vertical' else cbar.ax.set_xlabel(clabel)
        cbar.ax.set_title(ctitle)
        if ctick_draw:
            cbar.ax.tick_params(axis = 'y' if corientation == 'vertical' else 'x', which = "both", direction = 'out')
        if cticks is not None:
            cbar.set_ticks(cticks)
        if ctick_labels is not None:
            cbar.set_ticklabels(ctick_labels)
        if not cminorticks:
            cbar.ax.minorticks_off()
            
    # add title
    ax.set_title(title, fontsize = fontsize)
    
    # save the figure
    if filename is not None:
        plt.savefig(filename, dpi = dpi, bbox_inches = 'tight')
        
    # return the plotted object
    if return_object:
        return cs
        
        
def map_scatter(df, var, drop_nan = True, s = 12, edgecolor = "black", edgewidth = 0.3,
                projection = ccrs.Robinson(25), 
                xmin = -9.375, xmax = 36.875, ymin = 29.875, ymax = 71.875,
                cmap = plt.get_cmap("seismic_r", 15), vmin = None, vmax = None, norm = None,
                cbar = {"show": True}, borders = False,
                line_width = 0.0, line_color = "gainsboro", fill_color = "gainsboro",
                grid_width = 0.3, grid_delta = 10, resol = "110m",
                return_object = False, return_fig = False, title = None, fontsize = 10,
                figsize = (3,3), dpi = 150, ax = None, filename = None):
    
    """
    Plot scatter points on a map.
    
    :param pd.dataframe df: DataFrame with lat, lon, var columns.
    :param string var: Variable to plot (column name of df).
    :param bool drop_nan: Don't plot missing values.
    :param float s: Size of the scatter points.
    :param string edgecolor: Edgecolor of the scatter points.
    :param float edgewidth: Width of the contour around the scatter points.
    :param ccrs.proj projection: Projection of the map. Defaults to a good value for Europe.
                                 Use ccrs.Orthographic(-10, 45) to plot a globe.
    :param int xmin/xmax/ymin/ymax: Corners of the domain. Defaults to Europe.
    :param string cmap: Colormap for the plot.
    :param float vmin/vmax: Range for the colormap.
    :param matplotlib.color.norm norm: Norm for the colormap.
    :param dict cbar: Settings for the colorbar. Must contain key "show". Optional keys are "orientation", 
                      "extend", "label", "title", "ticks", "tick_labels", "minorticks".
    :param float line_width: Linewidth used for e.g. coastlines, edges of lakes, ...
    :param string line_color: Color of e.g. coastlines.
    :param string fill_color: Color of water.
    :param grid_width: Linewidth of the grid. Choose 0 for no grid.
    :param float grid_delta: Distance between grid lines (in degrees).
    :param str resol: Resolution of the coastlines. One of '10m', '50m', or '110m'.
    :param return_object: Whether to return the plotted object (can be useful when manually plotting one
                          colorbar for several panels, for example).
    :param string title: Title of the axis.
    :param float fontsize: Size of the font for the title.
    :param tuple figsize: Dimensions of the figure (if ax is None).
    :param float dpi: Dpi of the figure (if ax is None).
    :param matplotlib.ax ax: Axis on which to plot the map. Choose None to create new axis.
    :param string filename: Filename of the figure. Choose None if it shouldn't be saved.
    """
    
    # if no existing axis is provided: create a new figure
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = plt.subplot(111, projection = projection)
        
    # scatterplot
    sel = ~np.isnan(df[var]) if drop_nan else [True for _ in range(len(df))]                     
    cs = ax.scatter(df["lon"][sel], df["lat"][sel], c = df[var][sel], 
                    transform = ccrs.PlateCarree(), zorder = 100,
                    s = s, edgecolor = edgecolor, linewidth = edgewidth, 
                    cmap = cmap, norm = norm, vmin = vmin, vmax = vmax)
    
    # add coastlines
    ax.coastlines(resolution = resol, linewidth = line_width, rasterized = True)
    
    # add lakes and oceans
    lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', scale = resol, 
                                                edgecolor = line_color, linewidth = line_width,
                                                facecolor = fill_color)
    ax.add_feature(lakes, zorder = 0, rasterized = True)
    ax.add_feature(cartopy.feature.OCEAN.with_scale(resol), facecolor = fill_color, zorder = 0, rasterized = True)
    
    # draw borders
    if borders:
        ax.add_feature(cf.BORDERS, linewidth = grid_width, linestyle = "dashdot")
    
    # add grid
    gl = ax.gridlines(linewidth = grid_width, color = 'black', alpha = 0.25)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, grid_delta))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, grid_delta))
    
    # set the extent of the map
    if (xmin is not None) and (xmax is not None) and (ymin is not None) and (ymax is not None):
        ax.set_extent([xmin, xmax, ymin, ymax])
    
    # add colorbar
    if cbar["show"]:
        divider = make_axes_locatable(ax)
        
        # settings
        corientation = "vertical"
        cextend = "neither"
        clabel = None
        ctitle = None
        cticks = None
        ctick_labels = None 
        
        corientation = cbar["orientation"] if "orientation" in cbar else "vertical"
        cextend = cbar["extend"] if "extend" in cbar else "neither"
        clabel = cbar["label"] if "label" in cbar else None
        ctitle = cbar["title"] if "title" in cbar else None
        cticks = cbar["ticks"] if "ticks" in cbar else None
        ctick_draw = cbar["tick_draw"] if "tick_draw" in cbar else True
        ctick_labels = cbar["tick_labels"] if "tick_labels" in cbar else None
        cminorticks = cbar["minorticks"] if "minorticks" in cbar else True
        
        cax = divider.append_axes("right" if corientation == 'vertical' else "bottom", size = "5%", 
                                  pad = 0.05, axes_class = plt.Axes)
        cbar = plt.colorbar(cs, orientation = corientation, extend = cextend, cax = cax)
        cbar.solids.set_rasterized(True) # for good pdf rendering
        cbar.ax.set_ylabel(clabel) if corientation == 'vertical' else cbar.ax.set_xlabel(clabel)
        cbar.ax.set_title(ctitle)
        if ctick_draw:
            cbar.ax.tick_params(axis = 'y' if corientation == 'vertical' else 'x', which = "both", direction = 'out')
        if cticks is not None:
            cbar.set_ticks(cticks)
        if ctick_labels is not None:
            cbar.set_ticklabels(ctick_labels)
        if not cminorticks:
            cbar.ax.minorticks_off()
            
    # add title
    ax.set_title(title, fontsize = fontsize)
    
    # save the figure
    if filename is not None:
        plt.savefig(filename, dpi = dpi, bbox_inches = 'tight')
        
    # return the plotted object
    if return_object:
        return cs
    elif return_fig:
        return fig
