all:
	seq -f "psurface%02g.top" 0 99 | xargs make -k -j 8
# all:
# 	seq -f "results/%.0f.x" 180 1 200 | xargs make -k -j 8
# 	seq -f "results/%.0f.x" 100 100 800 | xargs make -k -j 8
# 	seq -f "results/%.0f.x" 100 20 800 | xargs make -k -j 8
# 	seq -f "results/%.0f.x" 100 5 800 | xargs make -k -j 8
# 	# seq -f "results/%.0f.x" 100 1 800 | xargs make -k -j 8
# 	seq -f "results/%.0f.ana" 100 5 800 | xargs make -k
# 	seq -f "results/%.0f.ana" 180 1 200 | xargs make -k -j 8
# 	make cat.ana
# results/%.x:
# 	python autoquench.py $* > $@
# %.ana: %.x analysis.py
# 	python analysis.py < $< > $@
# cat.ana: $(wildcard results/*.ana)
# 	cat $^ > $@

%.top: %.gro
	gmx x2top -f $< -o $@ -name cnt -noparam -pbc -ff oplsaa 2> $@.log


%-0.gro: %.tpr
	prev=`expr $* - 1`; prev=`printf '%05d' $$prev`; echo $$prev; \
	if [ -e $$prev.part*.trr ] ; \
		then echo 0 | gmx trjconv -f $$prev.part*.trr -s $*.tpr -o $*-0.tmp.gro; \
	elif [ -e $*.trr ] ; \
		then echo 0 | gmx trjconv -f $*.trr -s $*.tpr -o $*-0.tmp.gro; \
	fi
	python3 pbcfix.py < $*-0.tmp.gro > $*-0.gro

clean:
	-rm "#"*
distclean: clean
	-rm 000*
	-rm psur*.gro psur*.itp
