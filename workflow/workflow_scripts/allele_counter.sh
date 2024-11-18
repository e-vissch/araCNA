#!/bin/bash
# this is tog et battenberg to work, unfortunately don't know how to do it without hardcoding the paths, as this is fed as the executable
singularity exec --bind $read_mount:$read_mount:ro \
            --bind $work_dir:$work_dir:rw \
            $main_dir/singularity_images/ascatngs_4.5.0.sif alleleCounter "$@"
