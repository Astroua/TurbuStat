#!/usr/bin/python

'''
Utility Functions used for the simulation statistics suite


'''

import os

def fromfits(folder,keywords,header=True, verbose=False):
  '''
  Loads a fits file, prints size, bits per pixel, and array type
  Returns image and its header
  Generalized to load multiple fits files

  INPUTS
  ------

  folder - string
           folder to look for FITS files

  keywords - list
             image descriptor to look for in file name

  header - bool
           return header

  OUTPUTS
  -------

  img_dict - dictionary
             dictionary containing images and headers


  EXAMPLE
  -------
      keywords = ["centroid", "centroid_error", "moment0", "moment0_error"]
      folder = "image_location"
      images = fromfits(folder, keywords)

      images["centroid"][0] - centroid array
      images["centroid"][1] - centroid header

  '''

  from astropy.io.fits import getdata

  files = [x for x in os.listdir(folder) if x[-4:]=="fits"]

  img_dict = {}
  for img in files:

    path = "".join([folder,"/",img])

    match = find_word(path,keywords)
    try:
      pixelarray, hdr = getdata(path, header=header)
    except IOError:
      raise IOError("Corrupt FITS file: "+path)


    shape = pixelarray.shape

    ## Couldn't think of a clever check for the original cube, so just check the limiting cases
    if not match and len(shape)==3 and "cube" in keywords:
      match = "cube"

    if not match:
      print "No match for %s" % (img)
    else:
      img_dict[match] = [pixelarray,hdr]

      if verbose:
        if len(shape)==3:
          print "Shape : (%i,%i,%i)" % (shape[0],shape[1],shape[2])
        elif len(shape)==2:
          print "Shape : (%i,%i)" % (shape[0],shape[1])
        elif len(shape)==4:
          pixelarray = pixelarray[0,:,:,:]
          del hdr["NAXIS4"]
          print "4D array converted to 3D with shape (%i,%i,%i)" % (pixelarray.shape[0],pixelarray.shape[1],pixelarray.shape[2])
        print "BITPIX : %s" % (hdr["BITPIX"])

    match = None

  if len(keywords)==1:
    return pixelarray,hdr # No need for a dictionary with only one file
  else:
    if verbose:
      print "These are the image keywords %s" % (keywords)
    return img_dict

def find_word(text,search):
   '''
   For use in from fits. Finds the common word (by splitting at .) between a string (file name) and list of keywords
   keyword MUST be separated by periods to work!!
   '''

   dText = {}
   #dSearch = {}

   dText = text.split(".")

   found_word = 0
   try:
    match = list(set(dText) & set(search))[0]
    found_word = len([match])
   except IndexError:
    return False

   if found_word>1:
      print "Detection of two common words???"

   if found_word == 1:
      return match
   else:
      return False

def append_to_hdf5(filename, new_data, col_label):
  '''

  This function appends a new column onto pre-existing data frames.
  It allows missing comparisons to be added.

  Parameters
  **********

  filename : str
             File containing pre-existing data.

  new_data : array
             Array with shape (number of stats, timesteps)

  col_labels : str
               Label to be added for new column

  '''
  from pandas import HDFStore, concat

  store = HDFStore(filename)

  for i, key in enumerate(store.keys()):
    df = store[key]
    df = concat([df,new_data[i,:]])
    store[key] = df

  store.close()

