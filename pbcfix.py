from gromacs import read_gro, write_gro, compose, decompose
import sys
import numpy as np

for frame in read_gro(sys.stdin):
    mols = decompose(frame)
    cell = frame["cell"]
    celli = np.linalg.inv(cell)
    for water in mols["SOL"]:
        com = None
        for atom in water:
            if com is None:
                com = atom[1]
            else:
                d = atom[1] - com
                d -= np.floor(d @ celli + 0.5) @ cell
                atom[1] = com + d
    frame = compose(mols, cell)
    write_gro(frame, sys.stdout)
