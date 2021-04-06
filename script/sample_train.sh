#!/bin/bash

cd ..

./main.tsk \
	--task train \
	-i ./movielens/ratings.csv \
	-m ./movielens/movies.csv \
	--thread 20

