#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Directory containing the files
DIR="$1"

# Loop through all files in the directory
for file in "$DIR"/*; do
    # Ensure it's a file and not a directory
    if [ -f "$file" ]; then
        # Print the number of lines in the file (trimming whitespace)
        echo "Number of lines in $file: $(wc -l < "$file" | tr -d '[:space:]')"
        
        # Print the contents of the file
        cat "$file"
        
        # Ensure there's a newline before the dashed line
        echo
        
        # Print a horizontal dashed line for visual separation
        echo "-------------------------------------------------------"
    fi
done

