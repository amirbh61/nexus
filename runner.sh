#!/bin/bash

output_macro_Mfolder="/gpfs0/arazi/users/amirbenh/Resolving_Power/nexus/SquareFiberDatabaseExpansion2/"
sub_dir1="Geant4_PSF_events"
sub_dir2="Geant4_Kr_events"

max_jobs=5000
username="amirbenh"

# Loop over all geometry folders
for geometry_folder in $(find "$output_macro_Mfolder" -mindepth 1 -maxdepth 1 -type d)
do
  while true; do
    job_count=$(qstat | grep -E '^[0-9]' | grep "$username" | wc -l)  # Check the number of jobs in the queue
    if [ "$job_count" -lt "$max_jobs" ]; then
      # If the number of jobs is less than the maximum, submit a new job

      # Check sub_dir1 is defind. if so, send jobs to queue
      if [[ -n "$sub_dir1" ]]; then
        qsub -q arazi.q -o ~/Resolving_Power/nexus/SquareFiberCluster_outputs -e ~/Resolving_Power/nexus/SquareFiberCluster_errors job.sh "$geometry_folder" "$sub_dir1"
      fi

      # Check sub_dir2 is defind. if so, send jobs to queue
      if [[ -n "$sub_dir2" ]]; then
        qsub -q arazi.q -o ~/Resolving_Power/nexus/SquareFiberCluster_outputs -e ~/Resolving_Power/nexus/SquareFiberCluster_errors job.sh "$geometry_folder" "$sub_dir2"
      fi
      echo "$geometry_folder" >> geometries_sent_to_cluster.txt
      break
    else
      # If the maximum number of jobs is reached, wait for a while before checking again
      sleep 60
    fi
  done
# break # send only first geometry
done
