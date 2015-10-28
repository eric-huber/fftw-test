IDIR =./

CC=g++
CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -march=native -std=c++11 -I$(IDIR)

#For valgrind:
#CFLAGS=-fopenmp -g -Wall -ansi -O3 -DNDEBUG -fomit-frame-pointer -fstrict-aliasing -ffast-math -msse2 -mfpmath=sse -I$(IDIR)

a:=$(shell which icpc 2>&1 | tail -c5)
ifeq ($(a),icpc)
CC=icpc
CFLAGS=-O3 -ansi-alias -malign-double -fp-model fast=2 -openmp -I$(IDIR)
endif

MAKEDEPEND=$(CFLAGS) -O0 -M -DDEPEND
#LDFLAGS=-lfftw3 -lfftw3_threads -lm
LDFLAGS=-lboost_program_options -lfftw3 -lfftw3_omp -lm

vpath %.cc ./

FILES=fftw-test

FFTW=fftw++
EXTRA=$(FFTW) convolution
ALL=$(FILES) $(EXTRA)

all: $(FILES)

icpc: all
	CC=icpc

fftw-test: main.o $(FFTW:=.o)
	$(CC) $(CFLAGS) main.o $(FFTW:=.o) $(LDFLAGS) -o fftw-test

clean:  FORCE
	rm -rf $(ALL) $(ALL:=.o) $(ALL:=.d) fft*.txt

.SUFFIXES: .c .cc .o .d
.cc.o:
	$(CXX) $(CFLAGS) $(INCL) -o $@ -c $<
.cc.d:
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $(INCL) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(ALL:=.d)
endif

FORCE:
