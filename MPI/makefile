.SUFFIXES: .exe

MPCC	= mpicc
OPT		= -g
#OPT		= -O
LIBS	= -lm

.c.o:
	$(MPCC) $(OPT) -o $@ $< -c

all: mpi-allr.exe

mpi-allr-driver.exe: mpi-allr-driver.o mpi-allr.o
	$(MPCC) $(OPT) -o $@ $^ $(LIBS)

clean::
	rm -f *.o *.exe *~

###
