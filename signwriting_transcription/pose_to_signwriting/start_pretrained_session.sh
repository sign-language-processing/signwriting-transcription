#!/bin/bash

# File to store the output
PRETRAIN_OUTPUT_FILE="output_for_pretrain.log"
PIPELINE_OUTPUT_FILE="output_for_pipeline.log"

# Run the run_bash.sh script, capture both stdout and stderr
# Display the output on the screen and write to the file
./pretrain.sh | tee -a $PRETRAIN_OUTPUT_FILE
# Run the run_bash.sh script, capture both stdout and stderr
# Display the output on the screen and write to the file
./fine_tune.sh | tee -a $PIPELINE_OUTPUT_FILE
```
