
import numpy as np
import turbustat.analysis as ta
import os
import subprocess
import sys
import shutil

'''
Runs the basic analysis pipeline, starting from the outputted HDF5 files
'''

# Specify path with results

path = sys.argv[1]
path = path.rstrip("/")

if path.split("/")[-1] == "HDF5_files":
    hdf5_path = path.copy() + "/"
    path = "/".join(path.split("/")[:-1]) + "/"
else:
    hdf5_path = path + "/HDF5_files/"
    path += "/"

design_matrix = sys.argv[2]

if design_matrix == "None":
    design_matrix = None

# Convert into combined csv files.

good_comparison = []

print "Converting to combined csv files."

try:
    ta.convert_format(hdf5_path, 0, face2=0, design=design_matrix)
    shutil.move(hdf5_path+"distances_0_0.csv", path)
    good_comparison.append("0_0")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 0, face2=1, design=design_matrix)
    shutil.move(hdf5_path+"distances_0_1.csv", path)
    good_comparison.append("0_1")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 0, face2=2, design=design_matrix)
    shutil.move(hdf5_path+"distances_0_2.csv", path)
    good_comparison.append("0_2")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 1, face2=0, design=design_matrix)
    shutil.move(hdf5_path+"distances_1_0.csv", path)
    good_comparison.append("1_0")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 1, face2=1, design=design_matrix)
    shutil.move(hdf5_path+"distances_1_1.csv", path)
    good_comparison.append("1_1")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 1, face2=2, design=design_matrix)
    shutil.move(hdf5_path+"distances_1_2.csv", path)
    good_comparison.append("1_2")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 2, face2=0, design=design_matrix)
    shutil.move(hdf5_path+"distances_2_0.csv", path)
    good_comparison.append("2_0")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 2, face2=1, design=design_matrix)
    shutil.move(hdf5_path+"distances_2_1.csv", path)
    good_comparison.append("2_1")
except StandardError as err:
    print err
try:
    ta.convert_format(hdf5_path, 2, face2=2, design=design_matrix)
    shutil.move(hdf5_path+"distances_2_2.csv", path)
    good_comparison.append("2_2")
except StandardError as err:
    print err

# Next convert the fiducial comparisons

for fil in os.listdir(hdf5_path):
    if os.path.isfile(hdf5_path+fil) and "comparisons" in fil:

        out_name = ta.convert_fiducial(hdf5_path+fil, return_name=True)

        shutil.move(out_name, path)

# Now make the distance plots.

print "Making distance plots."

ta.comparison_plot(path, comparisons=good_comparison,
                   out_path=path+"Distance Plots/")

# Run the R-script to fit the data to the model

# Must have 0_0 and 2_2 comparisons to run

if not "0_0" in good_comparison and not "2_2" in comparison_plot:
    raise StandardError("Model fitting requires 0_0 and 2_2 to be available.")

os.chdir(path)

print "Fitting model of given design."

subprocess.call(['Rscript', path+"FactorialAnalysis.R"])

# This should create two output tables of the whole dataset.

# Now run the metric validation

print "Running metric validation."

subprocess.call(['Rscript',
                 "~/Dropbox/Dropbox/code_development/Examples/noise_validation.r",
                 path, "100"])

subprocess.call(['Rscript',
                 "~/Dropbox/Dropbox/code_development/Examples/signal_validation.r",
                 path, "100"])

# Finally, create the model plots

print "Creating model plots."

execfile("~/Dropbox/code_development/Examples/effect_plots.py")

effect_plots("DataforFits.csv", "ResultsFactorial.csv", save=True,
             out_path=path+'Model Plots/')

map_all_results("ResultsFactorial.csv", save=True, normed=True,
                out_path=path+'Model Plots/')

print "Finished!"
