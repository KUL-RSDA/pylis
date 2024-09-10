import importlib
import pickle
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
import numpy as np

def reload(library):
    importlib.reload(library)
    
    
def pickledump(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file)
        
        
def pickleload(filename):
    with open(filename, "rb") as file:
        obj = pickle.load(file)
    return obj


def argmax_nd(array):
    """
    Return the indices of the maximum element in an N-D array.
    """
    return np.unravel_index(array.argmax(), array.shape)


def argmin_nd(array):
    """
    Return the indices of the minimum element in an N-D array.
    """
    return np.unravel_index(array.argmin(), array.shape)


def root_zone(dc, weights = [0.1, 0.3, 0.6]):
    """
    Compute the root-zone (soil moisture/temperature) based on a weighted average of the layers.
    Default weights work for Noah-MP output.
    """
    rz = 0
    
    for layer, weight in enumerate(weights):
        rz += weight*dc.sel(layer = layer + 1)
        
    return rz


def iqr(dc, dim = "time"):
    """
    Compute the IQR for a datacube along one or more dimensions.
    """
    return dc.quantile(q = 0.75, dim = dim) - dc.quantile(q = 0.25, dim = dim)


def autocorr(dc, n_lags, missing = "conservative"):
    """
    Compute the lagged autocorrelation over the time dimension of a data cube.

    :param xr.DataArray dc: Data cube object on which to compute the lagged autocorrelation.
    :param int n_lags: Number of lags to compute the autocorrelation over.
    :param str missing: How to treat missing values. For options and behavior, see
                        https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acf.html
    """

    # dimension sizes
    n_time, n_x, n_y = len(dc.time), len(dc.x), len(dc.y)

    # data cube with autocorrelations
    ac = xr.DataArray(
        data = np.ones((n_lags+1, n_x, n_y))*np.nan,
        dims = ("n_lags", "x", "y"),
        coords = {"n_lags": np.arange(n_lags+1), "lon": dc.lon, "lat": dc.lat}
    )
    
    # loop over all grid cells, skip those where nothing was assimilated (all missing values in case of innovations)
    with tqdm(total = n_x * n_y) as pbar:
        for x in range(n_x):
            for y in range(n_y):
                if np.sum(np.isnan(dc.sel(x = x, y = y))) == n_time:
                    pass
                else:
                    ac.loc[dict(x = x, y = y)] = sm.tsa.acf(dc.sel(x = x, y = y), nlags = n_lags, missing = missing)
                pbar.update(1)
            
    return ac


def count_obs(dc, dim = "time", zero_as_nan = True):
    """
    Count the number of observations in a data cube along one or more dimensions.
    """
    n_obs = np.isfinite(dc).sum(dim = dim)
    
    if zero_as_nan:
        n_obs = n_obs.where(n_obs > 0, other = np.nan)
        
    return n_obs


def get_grid_latlon(dc, lat, lon):
    """
    Return the grid indices of the grid cell corresponding to a (lat, lon) pair for data cube dc
    """

    return argmax_nd((dc.lat == lat) * (dc.lon == lon))


def latlon_sel(dc, lat, lon, return_indices = False):
    """
    Find the closest grid cell of xr.DataArray dc for a given lat and lon in case of a non-orthogonal grid (when the .sel(lat = ..., lon = ...) functionality won't work)
    """

    # TO DO: see Michel's comment under PR #7
    # Using GDAL's osr and ogr modules, one can create geographic coordinates (points), assign them to a specific coordinate reference system (CRS), and compute the distance.

    flat_lats = dc.lat.data.flatten()
    flat_lons = dc.lon.data.flatten()

    distances = np.sqrt((flat_lats - lat)**2 + (flat_lons - lon)**2)
    min_index = distances.argmin()

    x_index, y_index = np.unravel_index(min_index, dc.lat.data.shape)

    if return_indices:
        return int(x_index), int(y_index)

    return dc.sel(x = x_index, y = y_index)


def corr(x, y):
    """
    Compute the Pearson correlation for two numpy arrays x and y that may contain missing values
    """
  
    mask = np.isfinite(x) * np.isfinite(y)
    
    return np.corrcoef(x[mask], y[mask])[0, 1] if np.sum(mask) > 0 else np.nan