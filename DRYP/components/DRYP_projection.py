import pyproj as pp
import numpy as np
""" component to reproject dataset from the native format to
Landlab reference system when dataset have different reference
systems
WARING: Do not activat this component if the reference system
is the new defined
"""
#lat = rg.node_y.reshape(rg.shape)
#lon = rg.node_x.reshape(rg.shape)
#grid_shape = np.array(rg.shape)
#xaxis = rg.node_x[:grid_shape[1]]
#yaxis = np.linspace(np.min(rg.node_y), np.max(rg.node_y),num=grid_shape[0])

# change projection
# define new projection (output) #!with.PYPROJ.library
newPP = pp.Proj(f'+proj=laea +lat_0=5 +lon_0=20 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')

# define current projection (input)
oldPP = pp.Proj(proj='latlong',datum='WGS84')

# transform from old CRS to new CRS
old_new_pp = pp.Transformer.from_proj(oldPP, newPP)

# read dataset
#fpet = xr.open_mfdataset(fname_pet)

def reproject_dataset(data, keys):
	""" reproject netcdf files to a new reference system
	the new reference sysitem has to be defined obove
	INPUT:
	-----
	data:	netcdf file read as xarray
	keys:	label of variables coordinata "longitude" and "latitude"
	OUTPUT:
	data:	dataset with reprojected coordinates
	"""
	# create raster grid of current projection
	x_old, y_old = np.meshgrid(data[keys[0]].values, data[keys[1]].values)
	
	# reproject data
	x_new, y_new = old_new_pp.transform(x_old, y_old, radians=False)
	
	# change to latitude longitude arrays
	data[keys[0]] = x_new[0][:]
	data[keys[1]] = y_new[:,0]

	return data

#fpet = fpet.interp(lat=(lat[:,0]), lon=(lon[0][:]))