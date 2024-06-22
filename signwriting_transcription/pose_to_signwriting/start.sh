#!/bin/bash

# File to store the output
OUTPUT_FILE="output.log"

# Run the run_bash.sh script, capture both stdout and stderr
# Display the output on the screen and write to the file
./run_bash.sh | tee -a $OUTPUT_FILE
