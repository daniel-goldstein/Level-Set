GCC := g++

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GccFLAGS = -fopenmp -O3

debug: GccFLAGS += -DDEBUG -g -Wall
debug: all

# The build target executable:
TARGET = lss

all: build

build: $(TARGET)

lss: lib/main.o lib/lss.o lib/imghandler.o
	$(GCC) $(GccFLAGS) $^ -o $@ 

lib/main.o: lib/main.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
lib/lss.o: lib/lss.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@ 
	
lib/imghandler.o: lib/imghandler.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
clean:
	$(RM) $(TARGET) *.o lib/*.o *.tar* *.core*
