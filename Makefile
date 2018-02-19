CC=mpicc
override CFLAGS +=-O3 -std=c99
LIBS=-lmpi -lm

cg: cg.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean: 
	rm -f cg
