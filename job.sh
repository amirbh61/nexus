#!/bin/bash

# This code runs a single job and is used inside runner.sh

source /gpfs0/arazi/projects/geant4.11.0.2-build/geant4make.sh
source /gpfs0/arazi/projects/root/bin/thisroot.sh
cd /gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/build

./build/nexus -b -n 462500 ${macro}
