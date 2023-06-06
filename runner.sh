#!/bin/bash

output_macro_Mfolder="/gpfs0/arazi/users/amirbenh/Resolving_power/nexus/SquareFiberMacrosAndOutput"

# Loop over all .config.mac files
for macro in $(find $output_macro_Mfolder -name "*.config.mac")
do
  # Submit each file as a job
  qsub -q fairshare.q -o ~/SquareFiberCluster_outputs -e ~/SquareFiberCluster_errors job.sh $macro
done
