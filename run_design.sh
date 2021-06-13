#!/bin/bash

#launch from design_runtime folder
  #contains resource folder
    #
  #contains results folder
    #create new folder named by datetime
    #put output in results folder if possible


input="./parameters.txt"
while IFS= read -r line
do
  echo "$line"
done < "$input"

#implement output path
python design.py

#read metrics file and pick best sequence(s)

#loop sequences
input="./output.txt"
while IFS= read -r line
do
  data = line.split(',')
  if tonumber(data[0]) > x
    python predict.py -path '/home/filename.npz' -seq data[1]
done < "$input"
