# Licensed under an MIT open source license - see LICENSE


'''
Script to create final output form of the HDF5 results files.
'''

import numpy as np
from pandas import HDFStore, DataFrame, concat, read_csv, Series
import os


def convert_format(path, design, face1, face2, output_type="csv",
                   parameters=None, decimal_places=8, append_comp=True):
    '''
    Takes all HDF5 files in given path comparing face1 to face2 and combines
    them into a single file.

    Parameters
    ----------
    path : str
        Path where files are located.
    design : str or pandas.DataFrame
             If str, assumes a 'csv' file.
    face1 : int
        Face of the cube.
    face2: int
        Face of the cube compared to.
    output_type : str, optional
           Type of file to output.
    parameters : list, optional
                 Contains column names of design that are the parameters
                 varied in the set. If None, all columns are appended to
                 the output file.
    decimal_places : int, optional
        Specify the number of decimal places to keep.
    append_comp : bool, optional
        Append on columns with fiducial numbers copy
    '''

    files = [path + f for f in os.listdir(path) if os.path.isfile(path + f)
             and "_"+str(face1)+"_"+str(face2)+"_" in f
             and "comparisons" not in f]
    files.sort()
    print "Files used: %s" % (files)

    if isinstance(design, str):
        design = read_csv(design)

    if isinstance(parameters, list):
        design_df = {}
        for param in parameters:
            design_df[param] = Series(design[param])
        design_df = DataFrame(design_df)
    else:
        design_df = design

    for i, f in enumerate(files):
        store = HDFStore(f)
        data_columns = {}
        # Get data from HDF5
        for key in store.keys():
            data = store[key].sort(axis=0).sort(axis=1)
            index = data.index
            mean_data = data.mean(axis=1)
            data_columns[key[1:]] = trunc_float(mean_data, decimal_places)
        store.close()

        # Add on design matrix
        for key in design_df:
            # can get nans if the file was made in excel
            design_df = design_df.dropna()
            design_df.index = index
            data_columns[key] = design_df[key]

        data_columns = DataFrame(data_columns)

        if append_comp:
            data_columns["Fiducial"] = \
                Series(np.asarray([i]*len(index)).T, index=index)
            data_columns["Designs"] = Series(index.T, index=index)

        if i == 0:  # Create dataframe
            df = data_columns
        else:  # Add on to dataframe
            df = concat([df, data_columns])

    filename = "distances_"+str(face1)+"_"+str(face2)

    if "Name" in df.keys():
        del df["Name"]

    if output_type == "csv":
        df.to_csv(path+filename+".csv")


def convert_fiducial(filename, output_type="csv", decimal_places=8,
                     append_comp=True, num_fids=5):
    '''
    Converts the fiducial comparison HDF5 files into a CSV file.

    Parameters
    ----------
    filename : str
        HDF5 file.
    output_type : str, optional
           Type of file to output.
    decimal_places : int, optional
        Specify the number of decimal places to keep.
    append_comp : bool, optional
        Append on columns with fiducial numbers copy
    num_fids : int, optional
        Number of fiducials compared.
    '''

    store = HDFStore(filename)
    data_columns = dict()
    for key in store.keys():
        data = store[key].sort(axis=1)
        mean_data = data.mean(axis=1)
        data_columns[key[1:]] = trunc_float(mean_data, decimal_places)
        comp_fids = store[key].index
    store.close()

    df = DataFrame(data_columns)

    if append_comp:
        fids = []
        for fid, num in zip(np.arange(0, num_fids-1), np.arange(num_fids-1, 0, -1)):
            for _ in range(num):
                fids.append(fid)

        df["Fiducial 1"] = Series(np.asarray(fids).T, index=df.index)
        df["Fiducial 2"] = Series(comp_fids.T, index=df.index)

    output_name = "".join(filename.split(".")[:-1]) + "." + output_type

    df.to_csv(output_name)


@np.vectorize
def trunc_float(a, places=8):
    '''
    Round a float, then truncate it.
    '''

    a_round = np.round(a, places)

    slen = len('%.*f' % (places, a_round))
    a_round = str(a_round)[:slen]

    return float(a_round)
