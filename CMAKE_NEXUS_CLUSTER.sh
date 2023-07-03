cmake -DGSL_ROOT_DIR=/usr -DHDF5_ROOT=/usr -DCMAKE_INSTALL_PREFIX=/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus -S . -B /gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/build
cd build
cmake --build . --target install -j 20
