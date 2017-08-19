
def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['data/*.fits', 'data/*.npz',
                                            'coveragerc']
    }
