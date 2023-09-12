#!/bin/bash

for randomSeed in 1;do
    for growingCriteria in QBC;do # rfMaxUncertainty QBC xgbMaxUncertainty_ibug
        for dataset in jarvis22;do 
            for target in e_form bandgap;do

                outputDir="al/$growingCriteria/$dataset/$target/$randomSeed"
	
	            if [ ! -d $outputDir ]; then
	                mkdir -p $outputDir
		    else
		        echo "skipping $outputDir"
		        continue # skip
	            fi
	
	                python run_al.py --growingCriteria $growingCriteria \
	                    --dataset $dataset --target $target --outputDir $outputDir --randomSeed $randomSeed 
	    done
	done
    done
done

