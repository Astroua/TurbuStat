# Licensed under an MIT open source license - see LICENSE


'''
Utility Functions used for the simulation statistics suite


'''

import os


def fromfits(folder, keywords=None, header=True, verbose=False):
    '''
    Loads a fits file, prints size, bits per pixel, and array type
    Returns image and its header
    Generalized to load multiple fits files

    Parameters
    ----------

    folder : str
        Folder to look for FITS files.

    keywords : list
        Image descriptions to look for in file name.

    header : bool, optional
        Sets whether the header is returned.

    verbose : bool, optional
        Prints array properties.

    Returns
    -------

    img_dict : dict
        Dictionary containing images and headers


    Example
    -------
        keywords = ["centroid", "centroid_error", "moment0", "moment0_error"]
        folder = "image_location"
        images = fromfits(folder, keywords)

        images["centroid"][0] - centroid array
        images["centroid"][1] - centroid header

    '''

    if keywords is None:
        keywords = {"centroid", "centroid_error", "integrated_intensity",
                    "integrated_intensity_error", "linewidth",
                    "linewidth_error", "moment0", "moment0_error", "cube"}

    from astropy.io.fits import getdata

    files = [x for x in os.listdir(folder) if x[-4:] == "fits"]

    img_dict = {}
    for img in files:

        path = "".join([folder, "/", img])

        match = find_word(path, keywords)
        try:
            pixelarray, hdr = getdata(path, header=header)
        except IOError:
            raise IOError("Corrupt FITS file: " + path)

        shape = pixelarray.shape

        # Couldn't think of a clever check for the original cube, so just check
        # the limiting cases
        if not match and len(shape) == 3 and "cube" in keywords:
            match = "cube"

        if not match:
            print "No match for %s" % (img)
        else:
            img_dict[match] = [pixelarray, hdr]

            if verbose:
                if len(shape) == 3:
                    print "Shape : (%i,%i,%i)" % (shape[0], shape[1], shape[2])
                elif len(shape) == 2:
                    print "Shape : (%i,%i)" % (shape[0], shape[1])
                elif len(shape) == 4:
                    pixelarray = pixelarray[0, :, :, :]
                    del hdr["NAXIS4"]
                    print "4D array converted to 3D with shape (%i,%i,%i)"\
                            % (pixelarray.shape[0], pixelarray.shape[1],
                               pixelarray.shape[2])
                print "BITPIX : %s" % (hdr["BITPIX"])

        match = None

    if len(keywords) == 1:
        return pixelarray, hdr  # No need for a dictionary with only one file
    else:
        if verbose:
            print "These are the image keywords %s" % (keywords)
        return img_dict


def find_word(text, search):
    '''
    For use in from fits. Finds the common word (by splitting at .) between a
    string (file name) and list of keywords.
    keyword MUST be separated by periods to work!!
    '''

    dText = {}
    dText = text.split(".")

    found_word = 0
    try:
        match = list(set(dText) & set(search))[0]
        found_word = len([match])
    except IndexError:
        return False

    if found_word > 1:
        print "Detection of two common words???"

    if found_word == 1:
        return match
    else:
        return False
