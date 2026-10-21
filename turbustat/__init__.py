from importlib import metadata

from ._astropy_init import __version__, test

try:
    __version__ = metadata.version("turbustat")
except metadata.PackageNotFoundError:
    __version__ = "unknown"
