"""
Read in a netCDF file and output the u and v data to json. Write one file per
time step for compatibility with the earth package:

    https://github.com/cambecc/earth

"""

import sys
import json
import time

import numpy as np

from netCDF4 import Dataset, num2date
from datetime import datetime


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

    Author
    ------
    Pierre Cazenave (Plymouth Marine Laboratory)

    """

    def __init__(self, file=None, uname=None, vname=None, xname=None, yname=None, tname=None, basedate=None, calendar=None, xdim=None, ydim=None, tdim=None, clip=None):
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
        self.y = ds.variables[self.config.xname][:]

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


        self.data = np.squeeze(ds.variables[var][
                dims[0][0]:dims[0][1],
                dims[1][0]:dims[1][1],
                dims[2][0]:dims[2][1],
                dims[3][0]:dims[3][1]
                ])

        self.time = ds.variables[self.config.tname][:]
        self.Times = []
        for t in self.time:
            self.Times.append(num2date(
                t,
                'seconds since {}'.format(self.config.basedate),
                calendar=self.config.calendar
                ))


class WriteJSON():
    """
    Write the Process object data to JSON in the earth format.

    Parameters
    ----------

    u, v : Process
        Process classes of data to export. Each time step will be exported to
        a new file.
    uconf, vconf : Config
        Config classes containing the relevant information. If omitted, assumes
        default options (see `Config.__doc__' for more information).

    """

    def __init__(self, u, v, uconf=None, vconf=None):
        if uconf:
            self.uconf = uconf
        else:
            self.uconf = Config()

        if vconf:
            self.vconf = vconf
        else:
            self.vconf = Config()

        self.write_json(u, v, uconf, vconf)

    def write_json(self, u, v, uconf, vconf):
        meta = {}
        meta['template'] = {
                'discipline':10,
                'disciplineName':'Oceanographic_products',
                'center':-4,
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
                'shapeName':'Earth spherical with radius = 6\n367,\n470 m',
                'scanMode':0,
                'nx':u.nx,
                'ny':u.ny,
                'lo1':u.x.min(),
                'la1':u.y.max(),
                'lo2':u.x.max(),
                'la2':u.y.min(),
                'dx':np.mean(np.median(np.diff(u.x[0, :]))),
                'dy':np.mean(np.median(np.diff(u.y[0, :])))
                }

        meta['u'], meta['v'] = {}, {}
        # Template is based on u data.
        meta['u']['header'] = meta['template']
        meta['v']['header'] = meta['template']
        meta['v']['header']['parameterNumber'] = 3
        meta['v']['header']['parameterNumberName'] = 'V_component_of_current'

        # Add the flattened data.
        meta['u']['data'] = u.data.flatten()
        meta['v']['data'] = v.data.flatten()

        # TODO:
        # Replace invalid values (9.96920997e+36) with Nones so they're
        # exported as nulls in the JSON file.

        return meta



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

    W = WriteJSON(u, v, uconf=uconfig, vconf=vconfig)


