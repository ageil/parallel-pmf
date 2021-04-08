# Parallel Probabilistic Matrix Factorization using C++

## About
Probablistic Matrix Factorization is a class of graphical models commonly used for recommender systems. This project provides a parallel implementation of a Gaussian matrix factorization model utilizing stochastic gradient ascent with no locking to obtain unbiased Maximum A Posteriori (MAP) estimates of the latent user preference and attribute vectors.

## Requirements & Prequisite libraries
* Boost >= 1.7.0
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) >=  3.3.9

## Installation
```bash
git clone https://github.com/ageil/parallel-pmf.git
cd parallel-pmf/
cmake .
make
```
To compile & run the unit tests:<br>
```bash
cmake .
make test
```

## Running options
```
Parameters for Probabilistic Matrix Factorization (PMF):
  -h [ --help ]             Help
  -i [ --input ] arg        Input file name
  -m [ --map ] arg          Item mapping file name
  --task arg                Task to perform
                             [Options: 'train', 'recommend']
  -o [ --output ] arg       Output directory
                              [default: current_path/results/]
  -k [ --n_components ] arg Number of components (k)
                             [default: 3]
  -n [ --n_epochs ] arg     Num. of learning iterations
                              [default: 200]
  -r [ --ratio ] arg        Ratio for training/test set splitting
                             [default: 0.7]
  --thread arg              Number of threads for parallelization
  --gamma arg               Learning rate for gradient descent
                              [default: 0.01]
  --std_theta arg           Std. of theta's prior normal distribution
                              [default: 1]
  --std_beta arg            Std. of beta's prior normal distribution
                              [default: 1]
  --user                    Recommend items for given user
  --item                    Recommend similar items for a given item
  --genre                   Recommend items for a given genre
  -s [--run_sequential]     Enable running fit model sequentially
  -l [--loss_interval] arg  Number of epochs between each loss computation. [default: 10]
```

## Quick start
Please refer to the sample running scripts for [training](script/sample_train.sh) and [recommendation](script/sample_recommend.sh)

## References
- Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. *Advances in neural information processing systems*, *20*, 1257-1264
- Niu, F., Recht, B., Ré, C., & Wright, S. J. (2011). Hogwild!: A lock-free approach to parallelizing stochastic gradient descent. arXiv preprint arXiv:1106.5730
- GroupLens Research (2021). MovieLens dataset. https://grouplens.org/datasets/movielens/
