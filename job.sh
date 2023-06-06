#!/bin/bash

# This code runs a single job and is used inside runner.sh

source /gpfs0/arazi/projects/geant4-build/share/Geant4-10.5.1/geant4make/geant4make.sh
source /gpfs0/arazi/projects/root/bin/thisroot.sh
cd /gpfs0/arazi/users/amirbenh/EL/build

./build/nexus -b -n 462500 ${macro}
