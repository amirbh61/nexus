#!/bin/bash
geometry_folder="/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/SquareFiberMacrosAndOutputsRandomFaceGen/ELGap=10mm_pitch=10mm_distanceFiberHolder=5mm_distanceAnodeHolder=2.5mm_holderThickness=10mm"

# Loop over all .init.mac files in the geometry folder
for macro in $(find "$geometry_folder" -name "*.init.mac")
do
  ./build/nexus -b -n 100000 "${macro}"
	job_count=$(find $geometry_folder -type f -name "*.txt" | wc -l)
	echo job number : $job_count
done




geometry_folder="/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/SquareFiberMacrosAndOutputsRandomFaceGen/ELGap=10mm_pitch=10mm_distanceFiberHolder=2mm_distanceAnodeHolder=2.5mm_holderThickness=10mm"

# Loop over all .init.mac files in the geometry folder
for macro in $(find "$geometry_folder" -name "*.init.mac")
do
  ./build/nexus -b -n 100000 "${macro}"
	job_count=$(find $geometry_folder -type f -name "*.txt" | wc -l)
	echo job number : $job_count
done



geometry_folder="/home/amir/Products/geant4/geant4-v11.0.1/MySims/nexus/SquareFiberMacrosAndOutputsRandomFaceGen/ELGap=10mm_pitch=10mm_distanceFiberHolder=-1mm_distanceAnodeHolder=2.5mm_holderThickness=10mm"

# Loop over all .init.mac files in the geometry folder
for macro in $(find "$geometry_folder" -name "*.init.mac")
do
  ./build/nexus -b -n 100000 "${macro}"
	job_count=$(find $geometry_folder -type f -name "*.txt" | wc -l)
	echo job number : $job_count
done
