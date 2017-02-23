#!/bin/bash

#############################################
#
# Execution script 
# ----------------
#
# by Julian Gutierrez
#
# Note: Run from main folder.
#############################################

#############################################
#
# Static configuration variables
#
#############################################

export OMP_NUM_THREADS=32
export OMP_NESTED=TRUE

# Set block sizes to be tested
#bslist="16"
bslist="4 8 16 32"
#bslist="2 4 8 16 32 64 128"

#############################################
#
# On the run configuration variables
#
#############################################

# Message
echo "*******************************"
echo "* Running Circle of the tests *"
echo "*******************************"
echo "By Julian Gutierrez"
echo ""

echo -n "Algorithm you want to run: "
read alg

echo -n "Name for the test to run: "
read testname

echo -n "How many iterations of the tests? Num: "
read iterations

echo -n "Run profiler for each test? (Takes longer) Yes/no: "
read profiler

if (( "$iterations" < 0 )) 
then
	iterations=1
elif (( "$iterations" > 5 )) 
then
	echo "Not saving images to avoid significant disk usage."
	answer="no"
else
	echo -n "Save images or not? Yes/no: "
	read answer
fi

echo -n "Which circle sizes do you want to test? (256 512 ...): "
read tests

# Code Versions
codes=$(ls algorithms/$alg)
#codes=$(ls algorithms/$algver | grep -v omp) 
#"cuda-basic cuda-basic-4 cuda-opt"

#############################################
#
# Script execution
#
#############################################

for test in $tests
do
	for pref in in out
	do
		echo "*********"
		echo "* Running circle-$pref $test test"
		echo "*********"
		
		#Set input files
		imagefolder="inputs/circle"

		#Set result folder
		result="results/$testname/circle-$pref-$test/"
		
		#Create inputs
		cd $imagefolder
		make -s clean
		make -s all
		./intensities $test
		./label-$pref $test
		cd -
			
		for code in $codes
		do
			#Set algorithm folder
			algfolder="algorithms/$alg/$code"

			#Remove previous results
			rm -rf $result/$code
			mkdir -p $result/$code

			echo "* Executing $code code for circle-$pref $test test"
			echo "***********"
			
			for bs in $bslist
			do
				#compile cuda code
				echo "* Compiling for block size $bs"
				cd $algfolder
				sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' src/config.h
				make -s clean 
				make -s all
				cd ../../../
				
				for (( i=1; i<=$iterations; i++ ))
				do			
					echo "* Running iteration $i"
					if [[ $alg == *"lss"* ]]; then
						$algfolder/lss \
						--image $imagefolder/circle-$test.intensities.pgm \
						--labels $imagefolder/circle-$test.label-$pref.pgm \
						--params $imagefolder/circle.params  > $result/$code/$bs-$i.log
					else 
						$algfolder/$alg \
						--image $imagefolder/circle-$test.intensities.pgm > $result/$code/$bs-$i.log
					fi
					if [ "$answer" == Yes ]; then
						mv result.ppm $result/$code/$bs-$i.ppm
					else
						rm result.ppm
					fi
				done
				#run profiler
				if [ "$profiler" == Yes ]; then
					if [[ $alg == *"lss"* ]]; then
						nvprof --metrics all $algfolder/lss \
						--image $imagefolder/circle-$test.intensities.pgm \
						--labels $imagefolder/circle-$test.label-$pref.pgm \
						--params $imagefolder/circle.params &> $result/$code/$bs.nvprof.log
					else 
						nvprof --metrics all $algfolder/$alg \
						--image $imagefolder/circle-$test.intensities.pgm &> $result/$code/$bs.nvprof.log
					fi
					rm result.ppm
				fi
			done
		done
	done
done

#Erase inputs
cd $imagefolder
make -s clean
cd -
