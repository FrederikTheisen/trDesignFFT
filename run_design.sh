#!/bin/bash

#launch from design_runtime folder
  #contains resource folder
    #
  #contains results folder
    #create new folder named by datetime
    #put output in results folder if possible

#implement something to wake up GPU
#implement output path
python design.py

#python script writes sequences to output.txt

#loop sequences
input="./output.txt"
while IFS= read -r line
do
    python predict.py -path '/home/filename.npz' -seq line
done < "$input"
