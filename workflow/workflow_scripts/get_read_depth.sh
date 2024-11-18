#!/bin/bash
# Assign command-line arguments to variables
bam_file="$1"
output_file="$2"
singularity_image_loc="$3"
read_mount="$4"
write_mount="$5"

declare -i i=0;
for r in $(samtools view $bam_file | cut -f3 | uniq | sed "s/*//")
do 
    singularity exec \
    --bind ${read_mount}:${read_mount}:ro \
    --bind ${write_mount}:${write_mount}:rw \
    ${singularity_image_loc} \
    samtools depth $bam_file -r $r | awk '{{print NR "\t" $0}}' > ${output_file}.tmp.depth.$r.out &
    pids[$i]=$!
    regs[$i]=$r
    i=$i+1
done
for pid in ${pids[*]}; do wait $pid; done
for r in ${regs[*]}; do cat ${output_file}.tmp.depth.$r.out >> $output_file; rm ${output_file}.tmp.depth.$r.out; done
