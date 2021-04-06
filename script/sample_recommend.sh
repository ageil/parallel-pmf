#/bin/bash

option=$1

cd ..

./main.tsk \
	--task recommend \
	-i movielens/ratings.csv \
	-m movielens/movies.csv \
	$option



	
