CC=gcc
MPICC=mpicc
CFLAGS=-O3 -std=gnu99 -fopenmp
MPI_CFLAGS=-O3 -std=gnu99 -DUSE_MPI
LIBS=-lm -llua -ldl

PORT=8080
NPROC=2
 
.PHONY: all run_local run_serial run_mpi stripped docs

# -----------------------------------
# Main build targets

all: wave1d.x wave1d_mpi.x

wave1d.x: wave1d.c driver_lua.c
	$(CC) $(CFLAGS) driver_lua.c wave1d.c -o wave1d.x $(LIBS)

wave1d_mpi.x: wave1d.c driver_lua.c
	$(MPICC) $(MPI_CFLAGS) driver_lua.c wave1d.c -o wave1d_mpi.x $(LIBS)

# -----------------------------------
# Run web server (for serving up glwave1d.html)

server:
	python -m SimpleHTTPServer $(PORT) 

# -----------------------------------
# Generate stripped assignment file from reference soluation

stripped:
	rm -rf strip
	mkdir strip
	cp Makefile README.md USAGE.md glwave1d.html *.lua *.c *.h strip
	awk -f strip.awk wave1d.c > strip/wave1d.c

# -----------------------------------
# Documentation generation

docs: doc/hw2.html doc/hw2.pdf
	cp doc/hw2.html ../../web/html
	cp doc/hw2.pdf ../../web/

doc/hw2.md: doc/wave_doc.md wave1d.h wave1d.c driver_lua.c
	make stripped
	cp doc/wave_doc.md $@
	ldoc -p pandoc -attribs '.c' wave1d.h strip/wave1d.c driver_lua.c >> $@

%.md: %.c
	ldoc -p pandoc -attribs '.c' -o $@ $^

%.html: %.md
	pandoc $< -s --toc -c '/pandoc.css' \
	        --mathjax --highlight-style pygments -o $@

%.pdf: %.md
	pandoc $< -s --toc -c pandoc.css \
	        --highlight-style pygments -o $@

# -----------------------------------
# Clean up

clean:	
	rm -f *.x *.o

realclean: clean
	rm -f u_plot.txt

