#!/usr/bin/env bash

# Takes a directory built with trainscripts builder
# and submits its batches to the cluster
source deactivate
source activate tensorflow
source /Users/akivalipshitz/Developer/netrain/jobsinit.sh                                

folder="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $folder
for arch in $( ls $folder/scripts/*.py ); do
    sbatch --array=1-1 $arch
done
