#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_wig_file> <output_wig_file>"
    exit 1
fi

# Assign command-line arguments to variables
file="$1"
wig_file=$file
output_file="$2"

# Read the line numbers of headers into an array
mapfile -t header_lines < <(grep -n "^fixedStep" "$file" | cut -d: -f1)

# Add the last line number of the file to the array
header_lines+=($(wc -l < "$file"))

# Initialize an array to store the line numbers to be removed
remove_lines=()

# Iterate over the array to process each header line number
for ((i=0; i<${#header_lines[@]}-1; i++)); do
    prev_header_line=${header_lines[i]}
    next_header_line=${header_lines[i+1]}
    
    # Calculate the number of lines between current and previous header
    line_count=$((next_header_line - prev_header_line - 1))
    
    # Check if exactly one line is present between headers
    if [ $line_count -eq 1 ]; then
        # Add the lines to be removed to the array
        echo $prev_header_line
        remove_lines+=($prev_header_line $((prev_header_line + 1)))
    fi
done

# Create a new file excluding the lines to be removed
awk 'NR == FNR {del[$1]; del[$2]; next} !(NR in del)' <(printf "%s\n" "${remove_lines[@]}") "$file" > "$output_file"

echo "Filtered file created: $output_file"