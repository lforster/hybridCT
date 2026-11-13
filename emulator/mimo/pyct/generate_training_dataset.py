import numpy as np
import xarray as xr
try:
    from emulator.mimo.pyct import utils
except ImportError:
    import utils
import os

def coarsen_resolution(dataset, res, coarse_res):
    """Return a coarsened dataset based on the original resolution `res` and
    the target coarse resolution `coarse_res`.

    Args:
        dataset (xarray.DataArray or xarray.Dataset): Input dataset.
        res (float): Original resolution of the dataset.
        coarse_res (float): Target coarse resolution.

    Returns:
        xarray.DataArray or xarray.Dataset: Coarsened dataset.
    """
    window_size = int(coarse_res/res)
    print(f"Coarsening dataset using window size {window_size}")
    coarse = dataset.coarsen(x=window_size, boundary='trim').mean().coarsen(y=window_size, boundary='trim').mean()
    return coarse


def get_projection_sortIndex(viewing_zenith_angle, delta_z, grid, grid_resolution):
    """ Get sort index to apply to original cloud image (registered to the ground)
    to obtain a projection to altitude delta_z [km] above ground.

    Args:
        viewing_zenith_angle: angle in degrees
        delta_z: altitude difference for projection, up is positive
        grid (list or np.nd-array): range of horizontal coordinates (1D) in direction of projection
        grid_resolution: resolution of horizontal grid
        verbose (bool): set to True to print debugging info

    Returns:
        sort_idx (int): index array to apply to data array to obtain projection
    """
    tan_factor = np.tan(np.deg2rad(-viewing_zenith_angle))
    shift = tan_factor * delta_z
    grid_new = grid + shift
    if shift < 0:
        mask = grid_new < grid.min()
        grid_new[mask] = (grid.max() + grid_new[mask]) % (grid.max() + grid_resolution)
    elif shift > 0:
        mask = grid_new > grid.max()
        grid_new[mask] = grid_new[mask] % (grid.max() + grid_resolution)
    # sort index along shifted and wrapped grid axis
    sort_idx = np.argsort(grid_new)
    return sort_idx


def project2com(da, cloud_height_avg, dx):
    """Project DataArray to the cloud center-of-mass.

    Args:
        da (xarray.DataArray): Input data to project.
        cloud_height_avg (float): Projection height.
        dx (float): Grid resolution.

    Returns:
        xarray.DataArray: Projected data.
    """
    data_proj = np.zeros_like(da.data)
    x_coord = da.sel(vza=0).x.data
    for ivza, vza in enumerate(da.vza.data):
        idx = get_projection_sortIndex(vza, cloud_height_avg, x_coord, dx)
        data_proj[:,:,ivza] = da.sel(vza=vza).data[idx,:]
    # create a copy of da but use projected data
    da_proj = da.copy(data=data_proj)
    return da_proj

def get_closest_index(value, x, dx=0.02):
    """Find the closest index in array `x` for a given value,
    within a specified tolerance.

    Args:
        value (float): The x-value to find the closest index for.
        x (numpy.ndarray or list): Array of values to search within.
        dx (float, optional): Tolerance value to define closeness. Defaults to 0.02.

    Returns:
        tuple: Tuple containing the closest index and the corresponding value in `x`.
    """
    diff = np.abs(value - x)
    # Find indices within the tolerance
    idx = np.where(diff <= dx)[0]
    # Select the index with the minimum difference
    closest_idx = idx[np.argmin(diff[idx])]
    # Access the value corresponding to the selected index
    selected_value = x[closest_idx]
    return closest_idx, selected_value

def wrap_around(dim_slice, data, dx=0.02, dim="x"):
    """Wrap around the dataset along a specified dimension.

    Args:
        dim_slice (slice): The range of values to slice.
        data (xarray.DataArray or xarray.Dataset): The dataset to wrap around.
        dx (float, optional): The grid spacing. Defaults to 0.02.
        dim (str, optional): The dimension to wrap along. Defaults to "x".

    Returns:
        xarray.DataArray or xarray.Dataset: The wrapped dataset.
    """
    dim_max = data[dim].data.max()
    dim_min = data[dim].data.min()
    start = dim_slice.start
    stop = dim_slice.stop
    if start < dim_min:
        idx, ival = get_closest_index(dim_min - start, data[dim].data, dx)
        # move columns to beginning of field in negative x-direction
        return data.roll(x=idx, roll_coords=False).sel(**{dim: slice(dim_min, stop-start)})
    elif stop > dim_max:
        idx, ival = get_closest_index(dim_max-start, data[dim].data, dx)
        # move columns to beginning of field in negative x-direction
        return data.roll(x=idx, roll_coords=False).sel(**{dim: slice(0, stop-start)})
    else:
        return data.sel(**{dim: dim_slice})

def get_cos_projection(vza, dx, x_center, data):
    """Apply cosine projection based on the viewing zenith angle (VZA).

    Args:
        vza (float): Viewing zenith angle in degrees.
        dx (float): Grid spacing in the x direction.
        x_center (xarray.DataArray): X-center of the cloud or data array.
        data (xarray.DataArray): Input dataset to project.

    Returns:
        xarray.DataArray: cos-projected dataset..
    """
    factor = np.cos(np.deg2rad(vza))
    dx_proj = dx * factor
    xmax = data.x.data.max()
    xmin = data.x.data.min()
    x_proj = np.arange(xmin, xmax+dx_proj, dx_proj)
    nx = len(data.x)
    nx_proj = len(x_proj)
    ny = len(data.y)
    # create new array - reduced by factor cos(VZA) in x-direction
    rad_proj = np.zeros((nx_proj, ny))
    # place x-center of data at the x-center of rad_proj
    center_slice = slice((nx_proj-nx)//2, (nx_proj-nx)//2+nx)
    rad_proj[center_slice] = data.data
    # correct x_proj to match center x value of AN view
    x_proj_corr = x_proj - x_proj[nx_proj//2+1] + x_center.data[len(x_center)//2+1]
    data_proj = xr.DataArray(rad_proj, coords={"x": x_proj_corr, "y": data.y.data, "vza": vza}, dims=(["x", "y"]))
    # interpolate to x grid from nadir view
    data_new = data_proj.interp(x=x_center)
    return data_new

def find_cosine_window(vza, x_range_an, dx):
    """Find the window for cosine projection based on viewing zenith angle (VZA).

    Args:
        vza (float): Viewing zenith angle in degrees.
        x_range_an (numpy.ndarray or list): Range of x values for analysis.
        dx (float): Grid spacing in the x direction.

    Returns:
        slice: A slice object representing the cosine window.
    """
    factor = np.cos(np.deg2rad(vza))
    size = x_range_an.max() - x_range_an.min()
    radius = size/2.
    x_center = x_range_an.min() + radius
    window = slice(x_center - radius/factor-dx, x_center+radius/factor+dx)
    return window

def map2tiles(data, tile_size, stride=None, offset_x=0, offset_y=0, dx=None, angles=[], multiview=False,
              wrap_y=True):
    """
    Split a 2D map into quadratic image tiles of size (tile_size, tile_size)
    with a defined step size (stride). Optionally, it handles wrapping the
    last tile around the domain boundary if needed. In multiview mode, it can
    generate tiles for multiple viewing angles.

    Args:
        data (np.ndarray or xarray.DataArray): Input 2D data array to be split into tiles.
            If multiview is enabled, it must be an `xarray.DataArray` containing multiple views.
        tile_size (int): Size of the square tiles in pixels.
        stride (int, optional): Step size for moving across the 2D data. Defaults to `tile_size`.
        offset_x (int, optional): Horizontal offset to start tiling. Default 0.
        offset_y (int, optional): Vertical offset to start tiling. Default 0.
        dx (float, optional): Grid resolution, required if multiview mode is enabled. Default None.
        angles (list, optional): List of viewing angles for multiview mode. Required if multiview is
            enabled. Defaults to an empty list.
        multiview (bool, optional): If `True`, generate tiles for multiple viewing angles. Default `False`.

    Returns:
        np.ndarray: A 3D array of tiles (or a 4D array if `multiview` is enabled).
                    The shape of the output array is `(num_tiles, tile_size, tile_size, [channels])`.

    Raises:
        ValueError: If `multiview` is enabled but no `angles` or `dx` is provided.

    Notes:
        - The function assumes periodic boundary conditions for wrapping around the edges of the domain.
        - For `multiview`, nadir and off-nadir views are extracted and stacked along an additional dimension.

    Example:
        >>> tiles = map2tiles(data, tile_size=32, stride=16, offset_x=0, offset_y=0)
        >>> multiview_tiles = map2tiles(data, tile_size=32, stride=16, angles=[0, 30, 60], multiview=True, dx=0.5)

    """
    if stride is None:
        stride = tile_size

    nx, ny = data.shape[:2]
    tiles_ = []
    # wrap_y=False (y-cropped regions): only tiles fully inside the crop, no periodic seam
    j_stop = ny-tile_size+stride if wrap_y else ny-tile_size+1
    for j in range(0, j_stop, stride):
        for i in range(0, nx-tile_size+stride, stride):
            data_ = data
            # find slice in x-axis
            x_start = i + offset_x
            x_stop = x_start + tile_size
            slx = slice(x_start, x_stop)
            # wrap around x
            if x_stop > nx:
                data_ = np.roll(data, nx-x_start, axis=0)
                slx = slice(0, tile_size)

            # find slice in y-axis
            y_start = j + offset_y
            y_stop = y_start + tile_size
            sly = slice(y_start, y_stop)
            # wrap around y
            if y_stop > ny:
                data_ = np.roll(data_, ny-y_start, axis=1)
                sly = slice(0, tile_size)

            if multiview:
                if not angles:
                    raise ValueError("Please provide list of viewing angles to stack.")
                if dx is None:
                    raise ValueError("Please provide grid resolution for multiview mode.")
                # select nadir view
                nadir = data.sel(vza=0)[slx, sly]
                # loop over off-nadir views and correct for periodic BCs
                cameras = []
                for vza in angles:
                    if vza == 0:
                        cameras.append(nadir.data)
                    else:
                        # same window for +/-vza since all 9 views are around cloud COM
                        x_slice = find_cosine_window(abs(vza), nadir.x.data, dx)
                        off_nadir = wrap_around(x_slice, data.sel(vza=vza), dx)[:,sly]
                        off_nadir_proj = get_cos_projection(vza, dx, nadir.x, off_nadir)
                        cameras.append(off_nadir_proj.data)
                multi_angle_tile = np.stack(cameras, axis=-1)
                tiles_.append(multi_angle_tile)
            else:
                tiles_.append(data_[slx, sly])
    if multiview:
        tiles = np.stack(tiles_, axis=0)
        # remove small negative values due to rounding effects in cos-projection
        n_below_zero = np.sum(tiles < 0)
        if n_below_zero > 0:
            print(f"found {n_below_zero} pixels < zero with extreme value {np.min(tiles)}. Mapping back to zero.")
            tiles[tiles < 0] = 0
    else:
        tiles = np.array(tiles_, dtype=float)
    return tiles


def tiles2map(tiles, nx, ny, stride, offset_x=0, offset_y=0, trim=0):
    """
    Reverse operation to map2tiles: Merges a list of tiles back into a 2D map.

    This function takes a list or array of tiles and merges them into a 2D map, assuming
    the same stride used in the `map2tiles` function. The `nx` and `ny` parameters define
    the shape of the original map, while `offset_x` and `offset_y` allow for optional
    adjustments of the map's origin. The `trim` parameter helps avoid artifacts by trimming
    a specified number of pixels from each tile.

    Args:
        tiles (array-like): List or array of image tiles to be merged into a 2D map.
        nx (int): The width of the original 2D map (number of columns).
        ny (int): The height of the original 2D map (number of rows).
        stride (int): The number of pixels to step when combining tiles.
        offset_x (int, optional): Offset to apply along the x-axis. Defaults to 0.
        offset_y (int, optional): Offset to apply along the y-axis. Defaults to 0.
        trim (int, optional): The number of pixels to trim from each tile to avoid artifacts. Defaults to 0.

    Returns:
        np.ndarray: The merged 2D map with dimensions `(ny, nx, channels)` (if channels exist).

    Notes:
        - The function assumes that the last axis of each tile represents viewing angles/channels,
          if applicable.
        - It is assumed that the stride and tile shape are consistent with those used
          in the `map2tiles` function.
    """
    n_tiles, tile_size_x, tile_size_y = tiles.shape[:3]

    n_views = 0
    if len(tiles.shape) > 3:
        n_views = tiles.shape[-1]

    if n_views > 0:
        data = np.zeros((nx, ny, n_views))
    else:
        data = np.zeros((nx, ny))
    k = 0
    for j in range(0, ny-tile_size_y+stride, stride):
        for i in range(0, nx-tile_size_x+stride, stride):
            tile = tiles[k]
            # find slice in x-axis
            x_start = i + offset_x
            x_stop = x_start + tile_size_x
            # define slice for tile: crop to trim, otherwise use entire tile
            slx_tile = slice(trim, tile_size_x - trim)
            # if x slice wraps around nx, crop tile to remaining rows
            if x_stop > nx:
                x_stop = nx + trim # will be subtracted later
                tile = tiles[k][:nx-x_start, :]
                # fit trim to border tile size
                if trim >= tile.shape[0]:
                    slx_tile = slice(None, None)
            # define x-slice of data map
            slx = slice(x_start+trim, x_stop-trim)
            # find slice in y-axis
            y_start = j + offset_y
            y_stop = y_start + tile_size_y
            # define slice for tile: crop to trim, otherwise use entire tile
            sly_tile = slice(trim, tile_size_y - trim)
            # if y slice wraps around ny, crop tile to remaining columns
            if y_stop > ny:
                y_stop = ny + trim # will be subtracted later
                tile = tile[:,:ny-y_start]
                # fit trim to border tile size
                if trim < tile.shape[1]:
                    sly_tile = slice(trim, None)
                else:
                    sly_tile = slice(trim, None)
            # define y-slice of data map
            sly = slice(y_start+trim, y_stop-trim)
            #print(slx, sly, data.shape, tile.shape)
            if n_views > 0:
                data[slx,sly,:] = tile[slx_tile, sly_tile, :]
            else:
                data[slx,sly] = tile[slx_tile, sly_tile]
            k+=1
    return data


def preprocess_global(config):
    od_tiles = []
    rad_tiles = []
    shape = None
    for input_fname, target_fname in zip(config['input_fname'], config['target_fname']):
        rad_tiles_single, od_tiles_single, *shape = preprocess_single_file(input_fname, target_fname, config)
        rad_tiles.append(rad_tiles_single)
        od_tiles.append(od_tiles_single)
    return np.vstack(rad_tiles), np.vstack(od_tiles), *shape 

def crop_region(da, region):
    """Restrict a DataArray to a y band given as fractions [f0, f1] of the y extent."""
    if region is None:
        return da
    ny = da.sizes["y"]
    y0, y1 = int(round(region[0] * ny)), int(round(region[1] * ny))
    return da.isel(y=slice(y0, y1))


def preprocess_single_file(input_fname, target_fname, config, region=None):
        """Tile one scene. ``region=[f0, f1]`` restricts tiles to a y band (fractions of the
        y extent, applied after projection / coarsening; no periodic wrap across the crop)."""
        od_tiles = []
        rad_tiles = []
        # Load and process input and target data
        print(f"Reading input data from {input_fname}")
        rad = xr.open_dataset(input_fname)[config['input_field']]
        print(f"Reading target data from {target_fname}")
        tau = xr.open_dataset(target_fname)[config['target_field']]

        # Project to cloud center-of-mass if altitude is provided
        if "cloud_height_avg" in config:
            print(f"Projecting input data from 0 to {config['cloud_height_avg']} km altitude.")
            rad_proj = project2com(rad, config['cloud_height_avg'], config['dx'])
            print(f"Projecting target data from 0 to {config['cloud_height_avg']} km altitude.")
            tau_proj = project2com(tau, config['cloud_height_avg'], config['dx'])
        else:
            rad_proj = rad
            tau_proj = tau
            if config['multiview']:
                raise NotImplementedError(f"multiview=True in config requires cloud_height_avg [km] to be provided, \
                        which will be used as projection center and should ideally coincide with cloud COM. \
                        LOSs of all views will intersect at this altitude.")

        # Coarse-grain training data if resolution provided in config
        res = config['dx']
        if "resolution" in config.keys():
            if config['resolution'] < config['dx']:
                raise NotImplementedError(f"Please provide resolution > native resolution {config['dx']}.")
            elif config['resolution'] > config['dx']:
                res = config['resolution']
                rad_proj = coarsen_resolution(rad_proj, config['dx'], config['resolution'])
                tau_proj = coarsen_resolution(tau_proj, config['dx'], config['resolution'])
            else:
                tau_proj = tau_proj
                rad_proj = rad_proj

        rad_proj = crop_region(rad_proj, region)
        tau_proj = crop_region(tau_proj, region)
        wrap_y = region is None
        shape = rad_proj.shape[:2]

        tag = os.path.splitext(os.path.basename(input_fname))[0] # dataset identifier tag for filename
        res_tag = int(res*1e3) # resolution tag in [m] for filename
        if config["multiview"]:
            print("Multiview processing: project tiles such that line-of-sight intersects with cloud center-of-mass and stack viewing angles as channels.")
            _rad_tiles = map2tiles(rad_proj, config['tile_size'], config['stride'], dx=res, angles=config['angles'], multiview=config["multiview"], wrap_y=wrap_y)
            _od_tiles = map2tiles(tau_proj, config['tile_size'], config['stride'], dx=res, angles=config['angles'], multiview=config["multiview"], wrap_y=wrap_y)
            print("od_tiles", _od_tiles.shape, _rad_tiles.shape, config['stride'], config['tile_size'])
        else:
            print("Single-view processing: save one dataset per requested viewing angle.")
            #for vza in config['angles']:
            _rad_tiles = map2tiles(rad_proj.sel(vza=config['angles']).data, config['tile_size'], config['stride'], wrap_y=wrap_y)
            _od_tiles = map2tiles(tau_proj.sel(vza=config['angles']).data, config['tile_size'], config['stride'], wrap_y=wrap_y)
            print(f"Generated {len(_rad_tiles)} radiance and optical thickness tiles, with shape {_rad_tiles[0].shape}")
        od_tiles.append(_od_tiles)
        rad_tiles.append(_rad_tiles)
        return np.vstack(rad_tiles), np.vstack(od_tiles), *shape


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file")
    args = parser.parse_args()
    with open(args.yaml) as f:
        config = yaml.safe_load(f)

    rad_tiles, od_tiles, *shape = preprocess_global(config)
    # save results to netCDF
    utils.save2dataset(rad_tiles, od_tiles, *shape, config)


if __name__ == '__main__':
    main()
