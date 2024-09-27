# split a multiframed gro file into gro files.

import gromacs
import sys

basename = sys.argv[1]
with open(basename + ".gro") as fr:
    for i, frame in enumerate(gromacs.read_gro(fr)):
        with open(f"{basename}{i:02d}.gro", "w") as fw:
            gromacs.write_gro(frame, fw)
