#/bin/bash

option=$1 # options: user; item; genre

cd ..

./main.tsk \
	--task recommend \
	-i movielens/ratings.csv \
	-m movielens/movies.csv \
	$option

	
