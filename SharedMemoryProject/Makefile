all: matrix.cpp
	icc -openmp matrix.cpp -o single -DSINGLE
	icc -openmp matrix.cpp -o double -DDOUBLE
	icc -openmp matrix.cpp -o triple -DTRIPLE

test: all
	./single
	./double
	./triple

clean:
	rm single double triple *.e* *.o*
