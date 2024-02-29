all:
	seq -f "results/%.0f.x" 100 100 1000 | xargs make -k -j 8
	seq -f "results/%.0f.x" 100 20 1000 | xargs make -k -j 8
	seq -f "results/%.0f.x" 100 5 1000 | xargs make -k -j 8
	# seq -f "results/%.0f.x" 100 1 1000 | xargs make -k -j 8
results/%.x:
	python autoquench.py $* > $@
