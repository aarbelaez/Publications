all: ubcsat

hfiles = src/algorithms.h \
	 src/mt19937p.h \
         src/mt19937ar.h \
         src/mylocal.h \
         src/reports.h \
         src/ubcsat-globals.h \
         src/ubcsat-internal.h \
         src/ubcsat-io.h \
         src/ubcsat-limits.h \
         src/ubcsat-lit.h \
         src/ubcsat-mem.h \
         src/ubcsat-time.h \
         src/ubcsat-triggers.h \
         src/ubcsat-types.h \
         src/ubcsat.h

cfiles = src/Cooperation.cpp \
	 src/adaptnovelty.c \
         src/algorithms.c \
         src/ddfw.c \
         src/derandomized.c \
         src/g2wsat.c \
         src/gsat-tabu.c \
         src/gsat.c \
         src/gwsat.c \
         src/hsat.c \
         src/hwsat.c \
         src/irots.c \
         src/mt19937p.c src/mt19937ar.c \
         src/mylocal.c \
         src/novelty+p.c \
         src/novelty.c \
         src/parameters.c \
         src/paws.c \
         src/random.c \
         src/reports.c \
         src/rgsat.c \
         src/rnovelty.c \
         src/rots.c \
         src/samd.c \
         src/saps.c \
         src/sparrow.c \
         src/ubcsat-help.c \
         src/ubcsat-internal.c \
         src/ubcsat-io.c \
         src/ubcsat-mem.c \
         src/ubcsat-reports.c \
         src/ubcsat-time.c \
         src/ubcsat-triggers.c \
         src/ubcsat-version.c \
         src/ubcsat.c \
         src/vw.c \
         src/walksat-tabu.c \
         src/walksat.c \
         src/weighted.c

ubcsat: $(hfiles) $(cfiles)
	mpic++ -DSHARE_INFO1 -DTLIMIT=300 -DRESTART_PNormW -Wall -O3 -g -o ubcsat $(cfiles) -lm 

clean:
	rm -f ubcsat sparrow2011
