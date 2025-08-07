# Created by Marc Winstel on 01.08.2025

import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    FileName = "Dirichlet_uniform"


    marc = np.genfromtxt("./Marc/" + FileName + ".dat", dtype =float )
    #marc2 = np.genfromtxt("./Marc/" + FileName + "2.dat", dtype =float)
    
    jan_x = np.genfromtxt("./Jan/" + FileName + ".dat", dtype =float, usecols=(0) )
    jan_y = np.genfromtxt("./Jan/" + FileName + ".dat", dtype =float, usecols=(1))
    jan2_x = np.genfromtxt("./Jan/" + FileName + "2.dat", dtype =float, usecols=(0) )
    jan2_y = np.genfromtxt("./Jan/" + FileName + "2.dat", dtype =float, usecols=(1))

    counts, bin_edges = np.histogram(marc, bins=200, range=(0, 2000))
    #counts2, bin_edges2 = np.histogram(marc2, bins=200, range=(0, 2000))
    # counts contains the number of entries in each bin
    # bin_edges contains the edges of the bins

    bin_edges = bin_edges + 5.0
    print(bin_edges)

    plt.scatter(bin_edges[:-1], counts, alpha=0.9, label="Marc", color ="blue")
    #plt.scatter(bin_edges2[:-1], counts2, alpha=0.9, label="Marc2", color ="green")

    plt.scatter(jan_x, jan_y, alpha=0.9, label="Jan", color ="red")
    plt.scatter(jan2_x, jan2_y, alpha=0.9, label="Jan2", color ="orange")
    plt.legend()
    plt.show()