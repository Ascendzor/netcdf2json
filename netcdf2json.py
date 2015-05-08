"""
Read in a netCDF file and output the u and v data to json. Write one file per
time step for compatibility with the earth package:

    https://github.com/cambecc/earth

"""

import os
import sys
import glob
import json
import time
import multiprocessing

import numpy as np

from netCDF4 import Dataset, num2date
from datetime import datetime
from scipy.interpolate import griddata


def main(f):
    """
    Function to run the main logic. This is mapped with the multiprocessing
    tools to run in parallel.

    """

    def interpolate(var, r, config, extranan=np.inf):
        """
        Interpolate the data in u.data and v.data onto a grid from 0 to 360 and
        -80 to 80.

        Parameters
        ----------
        var : Process
            Process class objects with the relevant data to interpolate.
        r : float
            Grid resolution for the interpolation.
        config : Config
            Config class with the various configuration parameters for the data
            in var.
        extranan : float
            An extra check for nonsense data. Set to np.inf by default (i.e.
            ignored).

        Returns
        -------
        vari : Process
            Updated class with the interpolated data.

        """

        # Move longitudes to 0-360 instead of -180 to 180.
        var.x[var.x < 0] = var.x[var.x < 0] + 360
        lon, lat = np.arange(0, 360, r), np.arange(-80, 80 + r, r)
        LON, LAT = np.meshgrid(lon, lat)
        X, Y, VAR = var.x.flatten(), var.y.flatten(), var.data.flatten()
        var.data = griddata((X, Y), VAR, (LON.flatten(), LAT.flatten()))
        var.data = np.reshape(var.data, (len(lat), len(lon)))
        var.data[np.isnan(var.data)] = config.nanvalue
        # Fix unrealistic values from the interpolation to the nanvalue.
        var.data[var.data > extranan] = config.nanvalue
        var.data[var.data < -extranan] = config.nanvalue
        # Update the metadata
        var.x, var.y = LON, LAT
        var.dx, var.dy = r, r
        var.nx, var.ny = len(lon), len(lat)
        del(lon, lat, LON, LAT, X, Y, VAR, r)

        return var


    print('File {} of {}'.format(f + 1, len(files['u'])))

    uconfig = Config(file=files['u'][f],
            calendar='noleap',
            clip={'depthu':(0, 1), 'time_counter':(0, 1)})
    vconfig = Config(file=files['v'][f],
            calendar='noleap',
            clip={'depthv':(0, 1), 'time_counter':(0, 1)})
    pconfig = Config(file=files['chl1'][f],
            uname='Chl1',
            calendar='noleap',
            clip={'deptht':(0, 1), 'time_counter':(0, 1)})
    try:
        u = Process(files['u'][f], uconfig.uname, config=uconfig)
    except:
        print('Warning: interpolation for {} failed.'.format(files['u'][f]))
        return
    try:
        v = Process(files['v'][f], vconfig.vname, config=vconfig)
    except:
        print('Warning: interpolation for {} failed.'.format(files['v'][f]))
        return
    try:
        p = Process(files['chl1'][f], pconfig.uname, config=pconfig)
    except:
        print('Warning: interpolation for {} failed.'.format(files['chl1'][f]))
        return

    r = 1 # native resolution, but on a sensible grid.
    u = interpolate(u, r, uconfig, extranan=100)
    v = interpolate(v, r, uconfig, extranan=100)
    p = interpolate(p, r, uconfig, extranan=100)

    uvstem = os.path.join(out, 'nemo', '{}-{}_{:04d}'.format(
            os.path.split(os.path.splitext(uconfig.file)[0])[-1],
            os.path.split(os.path.splitext(vconfig.file)[0])[-1],
            f + 1
            ))
    pstem = os.path.join(out, 'ersem', '{}-{:04d}'.format(
            os.path.split(os.path.splitext(pconfig.file)[0])[-1],
            f + 1
            ))

    # Write out the JSON of the UV data and the chlorophyll data.
    W = WriteJSON(u, v, uconf=uconfig, vconf=vconfig, fstem=uvstem)
    W = WriteJSON(p, uconf=pconfig, fstem=pstem)


class Config():
    """
    Class for storing netCDF configuration options.

    Parameters
    ----------
    file : str, optional
        Full paths to the netCDF file.
    uname, vname, xname, yname, tname : str
        Names of the u and v velocity component variable names, and the x,
        y and time variable names.
    basedate : str, optional
        The time to which the time variable refers (assumes time is stored as
        units since some date). Format is "%Y-%m-%d %H:%M:%S".
    calendar : str, optional
        netCDF calendar to use. One of `standard', `gregorian',
        `proleptic_gregorian' `noleap', `365_day', `360_day', `julian',
        `all_leap' or `366_day'. Defaults to `standard'.
    xdim, ydim, tdim : str, optional
        Names of the x, y and time dimensions in the netCDF files.
    clip : dict, optional
        Dictionary of the index and dimension name to extract from the
        netCDF variable. Can be multiple dimensions (e.g. {'time_counter':(0,
        100), 'depthu':0}).
    nanvalue : float, optional
        Specify a value to replace with null values when exporting to JSON.

    Author
    ------
    Pierre Cazenave (Plymouth Marine Laboratory)

    """

    def __init__(self, file=None, uname=None, vname=None, xname=None, yname=None, tname=None, basedate=None, calendar=None, xdim=None, ydim=None, tdim=None, clip=None, nanvalue=None):
        self.__dict = {}
        self.__set(file, 'file', str)
        self.__set(uname, 'uname', str)
        self.__set(vname, 'vname', str)
        self.__set(xname, 'xname', str)
        self.__set(yname, 'yname', str)
        self.__set(tname, 'tname', str)
        self.__set(basedate, 'basedate', str)
        self.__set(calendar, 'calendar', str)
        self.__set(xdim, 'xdim', str)
        self.__set(ydim, 'ydim', str)
        self.__set(tdim, 'tdim', str)
        self.__set(clip, 'clip', dict)
        self.__set(nanvalue, 'nanvalue', float)

    def __set(self, value, target_name, value_type):
        if value:
            actual = value
        else:
            actual = self.__default[target_name]
        self.__dict[target_name] = value_type(actual)

    def __file(self):
        return self.__dict['file']

    def __uname(self):
        return self.__dict['uname']

    def __vname(self):
        return self.__dict['vname']

    def __xname(self):
        return self.__dict['xname']

    def __yname(self):
        return self.__dict['yname']

    def __tname(self):
        return self.__dict['tname']

    def __basedate(self):
        return self.__dict['basedate']

    def __calendar(self):
        return self.__dict['calendar']

    def __xdim(self):
        return self.__dict['xdim']

    def __ydim(self):
        return self.__dict['ydim']

    def __tdim(self):
        return self.__dict['tdim']

    def __clip(self):
        return self.__dict['clip']

    def __nanvalue(self):
        return self.__dict['nanvalue']

    file = property(__file)
    uname = property(__uname)
    vname = property(__vname)
    xname = property(__xname)
    yname = property(__yname)
    tname = property(__tname)
    basedate = property(__basedate)
    calendar = property(__calendar)
    xdim = property(__xdim)
    ydim = property(__ydim)
    tdim = property(__tdim)
    clip = property(__clip)
    nanvalue = property(__nanvalue)

    # Set some sensible defaults. These are based on my concatenated netCDFs of
    # Lee's global model run (so NEMO, I guess).
    __default = {}
    __default['file'] = 'test_u.nc'
    __default['uname'] = 'vozocrtx'
    __default['vname'] = 'vomecrty'
    __default['xname'] = 'nav_lon'
    __default['yname'] = 'nav_lat'
    __default['tname'] = 'time_counter'
    __default['basedate'] = '1890-01-01 00:00:00'
    __default['calendar'] = 'standard'
    __default['xdim'] = 'x'
    __default['ydim'] = 'y'
    __default['tdim'] = 'time_counter'
    __default['clip'] = {'depth':(0, 1)}
    __default['nanvalue'] = 9.969209968386869e+36


class Process():
    """
    Class for loading data from the netCDFs and preprocessing ready for writing
    out to JSON.

    """

    def __init__(self, file, var, config=None):
        if config:
            self.config = config
        else:
            self.config = Config()

        self.__read_var(file, var)

    def __read_var(self, file, var):
        ds = Dataset(file, 'r')
        self.nx = len(ds.dimensions[self.config.xdim])
        self.ny = len(ds.dimensions[self.config.ydim])
        self.nt = len(ds.dimensions[self.config.tdim])

        self.x = ds.variables[self.config.xname][:]
        self.y = ds.variables[self.config.yname][:]

        # Sort out the dimensions.
        if self.config.clip:
            alldims = {}
            for key, val in list(ds.dimensions.items()):
                alldims[key] = (0, len(val))
            vardims = ds.variables[var].dimensions

            for clipname in self.config.clip:
                clipdims = self.config.clip[clipname]
                common = set(alldims.keys()).intersection([clipname])
                for k in common:
                    alldims[k] = clipdims
            dims = [alldims[d] for d in vardims]


        self.data = np.flipud(np.squeeze(ds.variables[var][
                dims[0][0]:dims[0][1],
                dims[1][0]:dims[1][1],
                dims[2][0]:dims[2][1],
                dims[3][0]:dims[3][1]
                ]))

        self.time = ds.variables[self.config.tname][:]
        self.Times = []
        for t in self.time:
            self.Times.append(num2date(
                t,
                'seconds since {}'.format(self.config.basedate),
                calendar=self.config.calendar
                ))

        ds.close()


class WriteJSON():
    """
    Write the Process object data to JSON in the earth format.

    Parameters
    ----------

    u, v : Process
        Process classes of data to export. Each time step will be exported to
        a new file. If v is None, only write u.
    uconf, vconf : Config
        Config classes containing the relevant information. If omitted, assumes
        default options (see `Config.__doc__' for more information).

    """

    def __init__(self, u, v=None, uconf=None, vconf=None, fstem=None):
        self.data = {}

        if uconf:
            self.uconf = uconf
        else:
            self.uconf = Config()

        if vconf:
            self.vconf = vconf
        else:
            self.vconf = Config()

        self.header = {}
        # The y data is complicated because NEMO has twin poles.
        self.header['template'] = {
                'discipline':10,
                'disciplineName':'Oceanographic_products',
                'center':-3,
                'centerName':'Plymouth Marine Laboratory',
                'significanceOfRT':0,
                'significanceOfRTName':'Analysis',
                'parameterCategory':1,
                'parameterCategoryName':'Currents',
                'parameterNumber':2,
                'parameterNumberName':'U_component_of_current',
                'parameterUnit':'m.s-1',
                'forecastTime':0,
                'surface1Type':160,
                'surface1TypeName':'Depth below sea level',
                'surface1Value':15,
                'numberPoints':u.nx * u.ny,
                'shape':0,
                'shapeName':'Earth spherical with radius = 6,367,470 m',
                'scanMode':0,
                'nx':u.nx,
                'ny':u.ny,
                'lo1':u.x.min().astype(float),
                'la1':u.y.max().astype(float),
                'lo2':u.x.max().astype(float),
                'la2':u.y.min().astype(float),
                'dx':u.dx,
                'dy':u.dy
                }

        if v:
            self.write_json(u, uconf, v=v, vconf=vconf)
        else:
            self.write_json(u, uconf)


    def write_json(self, u, uconf, v=None, vconf=None, fstem=None):

        self.data['u'], self.data['v'] = {}, {}
        # Template is based on u data.
        self.data['u']['header'] = self.header['template'].copy()
        # Can't use datetime.strftime because the model starts before 1900.
        date = datetime.strptime(str(u.Times[uconf.clip[uconf.tname][0]]), '%Y-%m-%d %H:%M:%S')
        self.data['u']['header']['refTime'] = '{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:06.3f}Z'.format(
                date.year, date.month, date.day, date.hour, date.minute, date.second
                )
        # Add the flattened data.
        self.data['u']['data'] = u.data.flatten().tolist()
        self.data['u']['data'] = [None if i == uconf.nanvalue else i for i in self.data['u']['data']]

        # Do the same for v if we have it.
        if v:
            self.data['v']['header'] = self.header['template'].copy()
            self.data['v']['header']['parameterNumber'] = 3
            self.data['v']['header']['parameterNumberName'] = 'V_component_of_current'
            date = datetime.strptime(str(v.Times[vconf.clip[vconf.tname][0]]), '%Y-%m-%d %H:%M:%S')
            self.data['v']['header']['refTime'] = '{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:06.3f}Z'.format(
                    date.year, date.month, date.day, date.hour, date.minute, date.second
                    )
            self.data['v']['data'] = v.data.flatten().tolist()
            self.data['v']['data'] = [None if i == vconf.nanvalue else i for i in self.data['v']['data']]

        with open('{}.json'.format(self.fstem), 'w') as f:
            f.write('[')
            for var in self.data.iterkeys():
                s = json.dumps(self.data[var])
                f.write(s)
                if count < len(self.data.keys()) - 1: f.write(',')
            f.write(']')


if __name__ == '__main__':

    files = {}
    files['u'] = '/local1/pmpc1408/Scratch/test_u.nc'
    files['v'] = '/local1/pmpc1408/Scratch/test_v.nc'

    uconfig = Config(file=files['u'],
            calendar='noleap',
            clip={'depthu':(0, 1), 'time_counter':(-2, -1)})
    vconfig = Config(file=files['v'],
            calendar='noleap',
            clip={'depthv':(0, 1), 'time_counter':(-2, -1)})
    u = Process(files['u'], uconfig.uname, config=uconfig)
    v = Process(files['v'], vconfig.vname, config=vconfig)

    # Reinterpolate the model outputs onto a third of a degree grid to match
    # the existing data.
    r = 1 / 3.0
    lon, lat = np.arange(0, 360, r), np.arange(-80, 80 + r, r)
    LON, LAT = np.meshgrid(lon, lat)
    X, Y, U, V = u.x.flatten(), u.y.flatten(), u.data.flatten(), v.data.flatten()
    X[X < 0] = X[X < 0] + 360
    u.data = griddata((X, Y), U, (LON.flatten(), LAT.flatten()))
    u.data = np.reshape(u.data, (len(lat), len(lon)))
    v.data = griddata((X, Y), V, (LON.flatten(), LAT.flatten()))
    v.data = np.reshape(v.data, (len(lat), len(lon)))
    u.data[np.isnan(u.data)] = uconfig.nanvalue
    v.data[np.isnan(v.data)] = vconfig.nanvalue
    # Fix unrealistic values from the interpolation to the nanvalue.
    u.data[u.data > 100] = uconfig.nanvalue
    v.data[v.data > 100] = vconfig.nanvalue
    u.data[u.data < -100] = uconfig.nanvalue
    v.data[v.data < -100] = vconfig.nanvalue
    # Shift the data to be 20-380 instead of 0-360. God knows why the code
    # needs this to work.
    xi = np.argmin(np.abs(lon - 20))
    u.data = np.hstack((u.data[:, xi:], u.data[:, :xi]))
    v.data = np.hstack((v.data[:, xi:], v.data[:, :xi]))
    LON = np.vstack((LON[xi:, :], LON[:xi, :]))
    LAT = np.vstack((LAT[xi:, :], LAT[:xi, :]))
    # Update the metadata. Add the crazy offset needed for the data.
    u.x, u.y = LON + 20, LAT
    v.x, v.y = LON + 20, LAT
    u.dx, u.dy, v.dx, v.dy = 4 * [r]
    u.nx, u.ny, v.nx, v.ny = len(lon), len(lat), len(lon), len(lat)
    del(lon, lat, LON, LAT, X, Y, U, V, r)

    W = WriteJSON(u, v, uconf=uconfig, vconf=vconfig)


