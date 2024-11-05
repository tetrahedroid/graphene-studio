from graphenestudio.gromacs import read_gro
import sys

# 原子数を読みとるだけのためにgraphene.groを読みこんでいる。
for frame in read_gro(sys.stdin):
    Natom = frame.residue_id.shape[0]
    break

# 原子の個数だけバネをつける。
print("""[ position_restraints ]
; 炭素を固定する。
;  i funct       fcx        fcy        fcz""")
for i in range(Natom):
    print(f"{i+1} 1 10000 10000 10000")
