#!/bin/bash

output_macro_Mfolder="/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/SquareFiberMacrosAndOutputs"

# Loop over all .config.mac files
for macro in $(find "$output_macro_Mfolder" -name "*.config.mac")
do
  # Submit each file as a job
  qsub -q fairshare.q -o ~/Resolving_Power/nexus/SquareFiberCluster_outputs -e ~/Resolving_Power/nexus/SquareFiberCluster_errors job.sh "$macro"
done
