#############################################
#
# Execution script 
# ----------------
#
# by Julian Gutierrez
#
# Note: Run from main folder.
#############################################

# Message
echo "******************************"
echo "* Recollecting data per test *"
echo "******************************"
echo "By Julian Gutierrez"
echo ""

echo -n "Which test data you want to recollect?: "
read answer

# Remove previous results
rm -f results/$answer/*summary.csv
rm -f results/$answer/*/*summary.csv

#############################################
#
# Script execution
#
#############################################

tests=$(ls results/$answer)

for test in $tests
do
	echo "*********"
	echo "* Running $test recollection"
	echo "*********"
	
	#Set result folder
	result="results/$answer/$test"
	
	#Set result file
	resultFile="$result/$test.summary.csv"
	
	echo "Test,Code,block size,Max Kernel,Min Kernel,Avg Kernel,Max GPU Time,Min GPU Time,Avg GPU Time" > $resultFile
	#echo "Code,block size,Min Kernel,Avg Kernel" > $resultFile

	codes=$(ls results/$answer/$test | grep -v .csv)
	for code in $codes
	do		
		echo "* Recollecting $code data"
		echo "***********"
		
		cd $result/$code
		
		# Grep all iterations and find Max, Min, Avg
		bslist=$(ls | grep "log" | grep -v nvprof | sed "s/\-.*//" | sort -un)
		for bs in $bslist
		do
			
			grep "Kernel Execution Time" * | grep -v nvprof | grep ".\+$bs\-" -v | grep "$bs\-" | awk '{print $4}' > Kresultlist

			#calculate max
			Kmax=$(cat Kresultlist | sort -nr | head -n 1)
			
			#calculate min
			Kmin=$(cat Kresultlist | sort -n | head -n 1)
			
			#calculate avg
			Kavg=$(cat Kresultlist | awk '{ total += $1; count++} END {print total/count}')
			
			grep "Total GPU Execution Time" * | grep -v nvprof | grep "$bs\-" | awk '{print $5}' > Tresultlist
			
			#calculate max
			Tmax=$(cat Tresultlist | sort -nr | head -n 1)
			
			#calculate min
			Tmin=$(cat Tresultlist | sort -n | head -n 1)
			
			#calculate avg
			Tavg=$(cat Tresultlist | awk '{ total += $1; count++} END {print total/count}')

			#Save to file
			echo "$test,$code,$bs,$Kmax,$Kmin,$Kavg,$Tmax,$Tmin,$Tavg" >> ../../../../$resultFile
			#echo "$code,$bs,$Kmin,$Kavg" >> ../../../../$resultFile
			
			#remove temp files
			rm Kresultlist Tresultlist
			#rm Kresultlist
		done
		
		cd ../../../../
	done

	#Create summary for whole test
	cat $resultFile >> results/$answer/temp.summary.csv
done

#Modify summary file to correct output
grep -m 1 "Test" results/$answer/temp.summary.csv > results/$answer/summary.csv
grep -v "Test" results/$answer/temp.summary.csv >> results/$answer/summary.csv
rm results/$answer/temp.summary.csv
