#!/bin/bash

exec_path=./pmf/bin/
exec_file=../main.tsk

if [[ ! -d $exec_path ]]; then
	mkdir $exec_path
fi

# copy the orig. C++ binary executable under package directory
if [[ -f $exec_file ]]; then
	cp $exec_file $exec_path
	pip install .
else
	echo "Binary executable 'main.tsk' doesn't exist, please compile the C++ program parallel-pmf first"
fi

