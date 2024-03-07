#!/bin/bash

# This code runs a single job and is used inside runner.sh
source /gpfs0/arazi/projects/geant4.11.0.2-build/geant4make.sh
cd /gpfs0/arazi/users/amirbenh/Resolving_Power/nexus

geometry_folder=$1
sub_dir=$2

target_path="${geometry_folder}/${sub_dir}"

# Loop over all .init.mac files in the geometry folder
for macro in $(find "$target_path" -name "*.init.mac")
do

  if [ "$sub_dir" == "Geant4_Kr_events" ]; then
    ./build/nexus -b -n 925000 "${macro}" # for Kr events
    echo "$macro" >> macros_sent_to_cluster.txt
  fi

  if [ "$sub_dir" == "Geant4_PSF_events" ]; then
    ./build/nexus -b -n 10000 "${macro}" # for PSF events
    echo "$macro" >> macros_sent_to_cluster.txt
  fi

done
