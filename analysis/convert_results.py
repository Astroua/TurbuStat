
'''
Script to create final output form of the HDF5 results files.
'''

from pandas import HDFStore, DataFrame, concat, read_csv, Series
import os

def convert_format(path, design, face1, face2, output_type="csv", parameters=None):
    '''
    Takes all HDF5 files in given path comparing face1 to face2 and combines
    them into a single file.

    Parameters
    **********

    path : str

    design : str or pandas.DataFrame
             If str, assumes a 'csv' file.

    face1 : int

    face2: int

    output_type : str, optional
           Type of file to output.

    parameters : list, optional
                 Contains column names of design that are the parameters varied in the set.
                 If None, all columns are appended to the output file.


    '''

    files = [path+f for f in os.listdir(path) if os.path.isfile(path+f) and str(face1)+"_"+str(face2) in f and f[:9]!="fiducial_"]
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
            index =  data.index
            mean_data = data.mean(axis=1)
            data_columns[key[1:]] = mean_data
        store.close()

        # Add on design matrix
        for key in design_df:
            design_df = design_df.dropna()  # can get nans if the file was made in excel
            design_df.index = index
            data_columns[key] = design_df[key]

        if i == 0:  # Create dataframe
            df = DataFrame(data_columns)
        else:  # Add on to dataframe
            data_columns = DataFrame(data_columns)
            df = concat([df,data_columns])

    filename = "distances_"+str(face1)+"_"+str(face2)

    if output_type == "csv":
        df.to_csv(path+filename+".csv")