import getopt
import pandas as pd
import numpy as np
import sys
from visualize import visualize

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["visu"])
        data = pd.read_csv("../ressources/data.csv")
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    for opt, arg in opts:
        if (opt == "-a"):
            alpha = float(arg)
        elif (opt == "-r"):
            rate = int(arg)
    for opt, arg in opts:
        if opt == '--visu':
            visualize(data)

if __name__ == "__main__":
    main(sys.argv[1:])
