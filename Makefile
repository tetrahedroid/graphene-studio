all:
	seq -f "results/%.0f.x" 180 1 200 | xargs make -k -j 8
	seq -f "results/%.0f.x" 100 100 800 | xargs make -k -j 8
	seq -f "results/%.0f.x" 100 20 800 | xargs make -k -j 8
	seq -f "results/%.0f.x" 100 5 800 | xargs make -k -j 8
	# seq -f "results/%.0f.x" 100 1 800 | xargs make -k -j 8
	seq -f "results/%.0f.ana" 100 5 800 | xargs make -k
	seq -f "results/%.0f.ana" 180 1 200 | xargs make -k -j 8
	make cat.ana
results/%.x:
	python autoquench.py $* > $@
%.ana: %.x analysis.py
	python analysis.py < $< > $@
cat.ana: $(wildcard results/*.ana)
	cat $^ > $@
