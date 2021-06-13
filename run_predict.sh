#!/bin/bash

input="./parameters.txt"
while IFS= read -r line
do
  echo "$line"
done < "$input"

python predict.py design_result.fasta
