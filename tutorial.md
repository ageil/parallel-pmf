## Tutorial - Parallel Probabilistic Matrix Factorization 

Anders Geil, Sol Park, Yinuo Jin

<b>Overview</b>

Probabilistic Matrix Factorization (PMF) is a popular class of graphical models commonly used for recommender systems. In this project, we provide a parallel implementation of Gaussian matrix factorization. This tutorial briefly covers the basics of PMF models, then delves into the technical details of how to set up and use our application; first for model fitting (section 3) and then for prediction/recommendation (section 4). Users experienced with PMF models may skip section 1, and go directly to the technical prerequisites in section 2.



<b>$\Large\S$ 1. Probablistic Matrix Factorization Basics</b>

Probabilistic Matrix Factorization models belong to a larger class of collaborative filtering models. As opposed to content-based filtering, these models rely on users’ interactions with a fixed set of items to provide recommendations. In a nutshell, PMF models assign to each user a latent (meaning unobserved, inferred from data) vector representing her preferences. Similarly, each item (e.g. a movie) is assigned a latent vector representing its attributes. We expect high ratings to occur for items whose attribute vectors are similar to a user’s preference vector.

To obtain the latent preference and attribute vectors, however, we need to first *learn* them from data. In this application, we therefore expect the dataset to take on a certain form. We will cover this in the next section.



<b>$\Large\S$ 2 Prerequisites</b>

In this project, we expect the data to come in the form of two csv file with (at least) 3 columns. The first file contains user-to-item *rating* information. The header line *must* have the column titles “userId”, “itemId”, and “rating” as the first three columns. Any columns after that will be ignored. The table below serves as an example of the expected format of a dataset:

| userId   | itemId   | rating   |
| -------- | -------- | -------- |
| 1        | 1        | 4.0      |
| 1        | 3        | 2.5      |
| $\vdots$ | $\vdots$ | $\vdots$ |

The second file contains the *supplementary* information for all the <b>itemId</b> appeared in the first file. Similar to the first csv file, we expect an exact format for the header in accordance to the example below:

| itemId   | itemName                | itemAttributes                 |
| -------- | ----------------------- | ------------------------------ |
| 1        | Toy Story (1995)        | Adventure\|Animation\|Children |
| 3        | Grumpier Old Men (1995) | Comedy\|Romance                |
| $\vdots$ | $\vdots$                | $\vdots$                       |

Before proceeding, please check that the following external dependencies are installed in addition to a working copy of C++17 or higher:

* Eigen (v3.3.9+)

* Boost (v1.73.0+)

* CMake (v3.20+)

  

<b>$\large\S$ 3. Getting started</b>

We provide a light-weighted command-line program compiled from our C++ project to apply PMF model on the dataset. This comes in with two steps: (1). we need to fit the PMF model to *learn* the latent features of each user and item; (2). we apply the model learned from the input dataset to perform *recommendations*.  This section will guide you through loading your datasets into the program. 

First, to instruct the program which task it will perform, use the option `--task` : `--task train` will fit the dataset to learn the PMF model,  and `--task test` will use learned model to make recommendations.

To load your *ratings* dataset, use the option `-i` or `--input` to input its file path; Similarly, to load your item's *supplementary* dataset with `-m` or `--map`. However, if you opt to use our provided [Movielens](./movielens) datasets, just pass in `--use_defaults` or `-d`.

In our first example, we will use the provided default datasets.

```bash
./main.tsk -d
```

Immediately, you will notice that the program will start to fit the model. The default behavior for the program is to train, since our model can't make reasonable recommendations without seeing actual data. However, we also support another mode for recommendations, and our`--task` option provides you the interface to switch between "training" mode (`--task train`) and "recommendation" mode (`--task recommendation`). For now, we will focus on the training mode. 



<b>$\Large\S$ 4. Fitting & tuning the model</b>

After familiarizing yourself with the fundamental arguments you need to read in datasets, let's try using it to fit our first model. The program's default behavior is to run the model fitting parallelization, as it's the most exciting feature we offer. But let's illustrate fitting the model sequentially first. For this section, we will go over various tuning parameters specifically related to the PMF model with sequential mode, and then we will go in depth in optimization with parallelization.

```bash
./main.tsk -d --run_sequential
```

Notice that the model prints out several different statements as the program is running. Let’s briefly discuss what each of them means. 

The first statement you'll likely notice is the "eposh" count. In machine learning, we refer to an epoch as a full pass over the dataset, i.e. an epoch ends when the model has seen (and learned from) each entry in the dataset exactly once. We use this to measure how far along the model is in the fitting procedure. We may adjust the total number of epochs the model is set to process by the command line parameter `n_epochs` (set to 200 by default) as follows:

```bash
./main.tsk -d --run_sequential --n_epochs 400
```

How do we know how long to train the model for? Well, we want to keep fitting the model as long as our latent variable vectors keep updating. Once the updates to our vectors become sufficiently small, we say the model has *converged*:  the “loss” starts to stabilize around a narrow range of values. It typically suggests the model has found the optimum parameter values for the latent vectors. Indeed, the "loss" is exactly the second statement we notice in the printout on our command line.

OK, you might see that our model converges in the provided example, but this doesn't always happen smoothly on another dataset. To deal that case, we'll introduce a few more parameters we could tune. First, if the loss changes slowly, we may want to increase the *learning rate*, simply by setting the `gamma` parameter in our program:

```bash
./main.tsk -d --run_sequential --gamma 0.1
```



<b>$\Large\S$ 5. Using the model for recommendations</b>

