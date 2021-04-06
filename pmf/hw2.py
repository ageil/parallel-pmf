import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import PMF
from eval import rmse, R2, topN
np.random.seed(2)

# load data (userId, movieId, rating)
ratings = np.loadtxt('movielens/ratings.csv', delimiter=',', skiprows=1)[:, :3]

# center ratings around 0
mid = np.mean(np.unique(ratings[:,2]))
ratings[:,2] = ratings[:,2] - mid

# split into train/test
idx = np.random.binomial(1, 0.7, size=len(ratings)).astype(bool)
ratings_train = ratings[idx,:]
ratings_test = ratings[~idx,:]


# 1. fit models on train data
# KS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
KS = [3]
models = dict()
for K in KS:
    print('K:', K)
    model = PMF(data=ratings_train, K=K, eta=(1.0, 1.0))
    losses, theta, beta = model.train(iters=200, gamma=1e-2)
    models[K] = [model, losses, theta, beta]


# 2. plot losses on train data (check convergence)
plt.figure(figsize=(12, 6))
for K in KS:
    model_losses = models[K][0].losses
    plt.plot([i*10 for i in range(len(model_losses))],
             model_losses,
             label='K={}'.format(K))
plt.xlabel('epoch')
plt.ylabel('$\\log p(\\theta, \\beta, x_{in})$')
plt.legend()
# plt.savefig('out/losses.png')
plt.show()


# 3. pick model with best pred log likelihood on test data
x_test = ratings_test[:, :2]
y_test = ratings_test[:, 2]

best_K_loss = -np.inf
best_K = None
for K in KS:
    model = models[K][0]
    loss = model.loss(ratings_test)
    if loss > best_K_loss:
        best_K_loss = loss
        best_K = K
print('Best K:', best_K)


# 4. evaluate R^2, RMSE, (topN precision/recall is slow)
# RMSE: root mean squared error
rmse_zero = rmse(y_test, 0)
rmse_ybar = rmse(y_test, np.mean(y_test))

for K in KS:
    model = models[K][0]
    y_hat = model.predict(x_test)
    rmse_yhat = rmse(y_test, y_hat)
    r2_yhat = R2(y_test, y_hat)

    NS = [10, 50, 100, 500]
    top = dict()
    for N in NS:
        prec, recall = topN(model, ratings_test, N=N)
        top[N] = (prec, recall)

    models[K].append(y_hat)  # idx 4
    models[K].append(rmse_yhat)  # idx 5
    models[K].append(r2_yhat)  # idx 6
    models[K].append(top)  # idx 7

    print('K={0}'.format(K))
    print('RMSE(y=0):\t\t', rmse_zero)
    print('RMSE(y=y_bar):\t\t', rmse_ybar)
    print('RMSE(y=y_hat):\t\t', rmse_yhat)
    print('R^2:\t\t', r2_yhat)
    for N in NS:
        prec, recall = models[K][7][N]
        print('Top N={} precision:', prec)
        print('Top N={} recall:', recall)


# 5. plot rmse performance on test data
rmse_scores = [models[K][5] for K in KS]

plt.figure(figsize=(12,6))
plt.bar(KS, rmse_scores)
plt.hlines(rmse_zero, 1.5, 10.5, 'tab:red', 'dotted', label='RMSE($0$)')
plt.hlines(rmse_ybar, 1.5, 10.5, 'tab:red', 'dashed', label='RMSE($\\bar{y}$)')
plt.text(10.5, rmse_zero-0.025, 'RMSE($0$)', ha='right', va='center')
plt.text(10.5, rmse_ybar-0.025, 'RMSE($\\bar{y}$)', ha='right', va='center')
plt.xlabel('K')
plt.ylabel('RMSE($\\hat{y}$)')
plt.ylim(0.8, 1.3)
# plt.savefig('out/rmse.png', dpi=300)
plt.show()


# 6. inspect similar item vectors
movies = pd.read_csv('movielens/movies.csv')

def getDegree(v1, v2):
    radians = np.arccos(np.dot(v1, v2))
    return np.degrees(radians)

def getSimilarItems(model, ref_item, num):
    ref_vec = model.beta[ref_item]
    degs = np.zeros((len(model.beta), 2))

    for i, (item, vec) in enumerate(model.beta.items()):
        if item != ref_item:
            degs[i, 0] = item
            degs[i, 1] = getDegree(ref_vec, vec)

    degs = pd.DataFrame(degs, columns=['movieId', 'degree'])
    most_similar = movies.merge(degs, on='movieId').sort_values('degree')[:num]
    return most_similar


sim_items = getSimilarItems(models[best_K][0], 10, 5)  # GoldenEye = 10
print(sim_items)
