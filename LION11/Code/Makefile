PNorm=RESTART_PNormW
Prob=RESTART_PROB
OUTPUT=csls

pnorm:  src/adaptnovelty.c src/algorithms.c src/algorithms.h src/ddfw.c src/derandomized.c src/g2wsat.c src/gsat.c src/gsat-tabu.c src/gwsat.c src/hsat.c src/hwsat.c src/irots.c src/mt19937ar.c src/mylocal.c src/mylocal.h src/novelty.c src/novelty+p.c src/parameters.c src/paws.c src/random.c src/reports.c src/reports.h src/rgsat.c src/rnovelty.c src/rots.c src/samd.c src/saps.c src/ubcsat.c src/ubcsat.h src/ubcsat-globals.h src/ubcsat-help.c src/ubcsat-internal.c src/ubcsat-internal.h src/ubcsat-io.c src/ubcsat-io.h src/ubcsat-limits.h src/ubcsat-lit.h src/ubcsat-mem.c src/ubcsat-mem.h src/ubcsat-reports.c src/ubcsat-time.c src/ubcsat-time.h src/ubcsat-triggers.c src/ubcsat-triggers.h src/ubcsat-types.h src/vw.c src/walksat.c src/walksat-tabu.c
	g++ -D$(PNorm) -finline-functions -fopenmp -Wno-write-strings -O3 -lm -o $(OUTPUT) src/Cooperation.cpp src/adaptnovelty.c src/algorithms.c src/derandomized.c src/ddfw.c src/g2wsat.c src/gsat.c src/gsat-tabu.c src/gwsat.c src/hsat.c src/hwsat.c src/irots.c src/mt19937p.c src/mylocal.c src/novelty.c src/novelty+p.c src/parameters.c src/paws.c src/random.c src/reports.c src/rgsat.c src/rnovelty.c src/rots.c src/samd.c src/saps.c src/ubcsat.c src/ubcsat-help.c src/ubcsat-internal.c src/ubcsat-io.c src/ubcsat-mem.c src/ubcsat-reports.c src/ubcsat-time.c src/ubcsat-triggers.c src/vw.c src/walksat.c src/walksat-tabu.c

prob: src/adaptnovelty.c src/algorithms.c src/algorithms.h src/ddfw.c src/derandomized.c src/g2wsat.c src/gsat.c src/gsat-tabu.c src/gwsat.c src/hsat.c src/hwsat.c src/irots.c src/mt19937ar.c src/mylocal.c src/mylocal.h src/novelty.c src/novelty+p.c src/parameters.c src/paws.c src/random.c src/reports.c src/reports.h src/rgsat.c src/rnovelty.c src/rots.c src/samd.c src/saps.c src/ubcsat.c src/ubcsat.h src/ubcsat-globals.h src/ubcsat-help.c src/ubcsat-internal.c src/ubcsat-internal.h src/ubcsat-io.c src/ubcsat-io.h src/ubcsat-limits.h src/ubcsat-lit.h src/ubcsat-mem.c src/ubcsat-mem.h src/ubcsat-reports.c src/ubcsat-time.c src/ubcsat-time.h src/ubcsat-triggers.c src/ubcsat-triggers.h src/ubcsat-types.h src/vw.c src/walksat.c src/walksat-tabu.c
	g++ -D$(Prob) -finline-functions -fopenmp -Wno-write-strings -O3 -lm -o $(OUTPUT) src/Cooperation.cpp src/adaptnovelty.c src/algorithms.c src/derandomized.c src/ddfw.c src/g2wsat.c src/gsat.c src/gsat-tabu.c src/gwsat.c src/hsat.c src/hwsat.c src/irots.c src/mt19937p.c src/mylocal.c src/novelty.c src/novelty+p.c src/parameters.c src/paws.c src/random.c src/reports.c src/rgsat.c src/rnovelty.c src/rots.c src/samd.c src/saps.c src/ubcsat.c src/ubcsat-help.c src/ubcsat-internal.c src/ubcsat-io.c src/ubcsat-mem.c src/ubcsat-reports.c src/ubcsat-time.c src/ubcsat-triggers.c src/vw.c src/walksat.c src/walksat-tabu.c

static_prob:
	make -f Makefile_n clean
	make -f Makefile_n all "version=$(Prob)" "OUTPUT=$(OUTPUT)"
static_pnorm:
	make -f Makefile_n clean
	make -f Makefile_n all "version=$(PNorm)" "OUTPUT=$(OUTPUT)"
clean:
	rm -rf *.o ubcsat

