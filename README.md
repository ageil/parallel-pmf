# A parallel implementation of Probabilistic Matrix Factorization in C++

##About
Brief introduction of Probablistic Matrix Factorization...

## Requirement & Prequisite libraries
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
```bash
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
```

## Quick start
Please refer to the sample running scripts [here](script/)

## References
To be completed
