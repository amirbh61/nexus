#!/bin/bash

# This code runs a single job and is used inside runner.sh
macro=$1
source /gpfs0/arazi/projects/geant4.11.0.2-build/geant4make.sh
export LD_LIBRARY_PATH=$G4INSTALL/lib:$LD_LIBRARY_PATH
export HDF5_DIR=/usr
export HDF5_LIB=/usr/lib64
export HDF5_INC=/usr/include
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

source /gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/scripts/nexus_setup.sh 

#export HDF=/usr/lib64/libhdf5.so
#export HDF5_LIB=/usr/lib64/
#export HDF5_LIB=/usr/lib64/libhdf5.so.103
#export HDF5_INC=/usr/include
cd /gpfs0/arazi/users/amirbenh/Resolving_Power/nexus

./build/nexus -b -n 462500 ${macro}
