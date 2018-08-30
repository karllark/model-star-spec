
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from modelstarspec.specmodel import FullSpecModel


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star")
    parser.add_argument("--path", help="path to star files",
                        default='./')
    parser.add_argument("--png", help="save figure as a png file",
                        action="store_true")
    parser.add_argument("--eps", help="save figure as an eps file",
                        action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file",
                        action="store_true")
    args = parser.parse_args()


    specmodel = StellarModels("T*v10_z*.dat",reddened_star)
