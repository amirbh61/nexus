#!/bin/bash

# This code runs a single job and is used inside runner.sh
source /gpfs0/arazi/projects/geant4.11.0.2-build/geant4make.sh
cd /gpfs0/arazi/users/amirbenh/Resolving_Power/nexus

geometry_folder=$1

# Loop over all .init.mac files in the geometry folder
for macro in $(find "$geometry_folder" -name "*.init.mac")
do
  ./build/nexus -b -n 462500 ${macro}
  echo "$macro" >> macros_sent_to_cluster.txt
done

