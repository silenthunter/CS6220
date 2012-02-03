all: matrix.cpp
	g++ -fopenmp matrix.cpp -o single -DSINGLE
	g++ -fopenmp matrix.cpp -o double -DDOUBLE
	g++ -fopenmp matrix.cpp -o triple -DTRIPLE

test: all
	./single
	./double
	./triple

clean:
	rm single double triple *.e* *.o*
