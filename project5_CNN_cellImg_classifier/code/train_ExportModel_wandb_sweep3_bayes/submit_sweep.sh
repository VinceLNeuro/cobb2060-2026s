#!/bin/bash

# Submit first job
JID=$(sbatch --parsable submit_job.sbatch) #just print the JID
echo "Job 1: $JID"

# Chain 19 more jobs sequentially
for i in $(seq 2 20); do
    JID=$(sbatch --parsable --dependency=afterany:$JID submit_job.sbatch)
    echo "Job $i: $JID"
    sleep 1
done

echo "Done. Monitor with: squeue -u $USER"

