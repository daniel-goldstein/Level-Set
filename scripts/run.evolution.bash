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

# Set block sizes to be tested
bslist="16"

#############################################
#
# On the run configuration variables
#
#############################################

# Message
echo "****************************"
echo "* Running All of the tests *"
echo "****************************"
echo "By Julian Gutierrez"
echo ""

alg="lss"
algver="lss"
testname="evol-lss-opt"

# Code Versions
code="cuda-opt"

#Create synthetic input
cd inputs/synthetic/
make -s clean
make -s all
make -s run
cd ../../

cd inputs/circle/
make -s clean
make -s all
make -s run
cd ../../

#############################################
#
# Script execution
#
#############################################


#Set algorithm folder
algfolder="algorithms/$algver/$code"

bs="16"

#compile cuda code
echo "* Compiling for block size $bs"
cd $algfolder
sed -i.bak 's/define BLOCK_TILE_SIZE [0-9]\+/define BLOCK_TILE_SIZE '"$bs"'/' lib/config.h
make -s clean 
make -s all
cd -

echo "Running Circle-in Cuda-Opt for evolution curve"
test="circle-in"
imagefolder="inputs/circle"
result="results/$testname/$test/"

rm -rf $result/$code
mkdir -p $result/$code

for i in 01 08 16 25
do
	$algfolder/$alg \
	--image $imagefolder/circle-1024.intensities.pgm \
	--labels $imagefolder/circle-1024.label-in.pgm \
	--params $imagefolder/circle.params --max_reps $i > $result/$code/$i.log

	mv result.ppm $result/$code/$i.ppm

done

echo "Running Circle-out Cuda-Opt for evolution curve"
test="circle-out"
imagefolder="inputs/circle"
result="results/$testname/$test/"

rm -rf $result/$code
mkdir -p $result/$code

for i in 01 15 30 40
do
	$algfolder/$alg \
	--image $imagefolder/circle-1024.intensities.pgm \
	--labels $imagefolder/circle-1024.label-out.pgm \
	--params $imagefolder/circle.params --max_reps $i > $result/$code/$i.log

	mv result.ppm $result/$code/$i.ppm

done

echo "Running Coins Cuda-Opt for evolution curve"
test="coins"
imagefolder="inputs/$test"
result="results/$testname/$test/"

rm -rf $result/$code
mkdir -p $result/$code

for i in 1 2 3 4
do
	$algfolder/$alg \
	--image $imagefolder/$test.intensities.pgm \
	--labels $imagefolder/$test.label.pgm \
	--params $imagefolder/$test.params --max_reps $i > $result/$code/$i.log

	mv result.ppm $result/$code/$i.ppm

done

echo "Running Fractal Cuda-Opt for evolution curve"
test="fractal"
imagefolder="inputs/$test"
result="results/$testname/$test/"

rm -rf $result/$code
mkdir -p $result/$code

for i in 01 20 40 60
do
	$algfolder/$alg \
	--image $imagefolder/$test.intensities.pgm \
	--labels $imagefolder/$test.label.pgm \
	--params $imagefolder/$test.params --max_reps $i > $result/$code/$i.log

	mv result.ppm $result/$code/$i.ppm

done

echo "Running Synthetic Cuda-Opt for evolution curve"
test="synthetic"
imagefolder="inputs/$test"
result="results/$testname/$test/"

rm -rf $result/$code
mkdir -p $result/$code

for i in 01 04 08 13
do
	$algfolder/$alg \
	--image $imagefolder/$test.intensities.pgm \
	--labels $imagefolder/$test.label.pgm \
	--params $imagefolder/$test.params --max_reps $i > $result/$code/$i.log

	mv result.ppm $result/$code/$i.ppm

done

echo "Running Tree Cuda-Opt for evolution curve"
test="tree"
imagefolder="inputs/$test"
result="results/$testname/$test/"

rm -rf $result/$code
mkdir -p $result/$code

for i in 01 12 24 36
do
	$algfolder/$alg \
	--image $imagefolder/$test.intensities.pgm \
	--labels $imagefolder/$test.label.pgm \
	--params $imagefolder/$test.params --max_reps $i > $result/$code/$i.log

	mv result.ppm $result/$code/$i.ppm

done

