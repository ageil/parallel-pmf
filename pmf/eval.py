import numpy as np

def rmse(y, y_hat):
    return np.sqrt(1/len(y) * np.sum((y - y_hat)**2))

def R2(y, y_hat):
    SSE = np.sum((y - y_hat)**2)
    TSS = np.sum((y - np.mean(y))**2)
    R2 = 1 - SSE/TSS
    return R2

def topN(model, ratings, N=10):
    liked = []
    hits = []
    for i in ratings[:, 0]:
        # get all items user_i liked
        i_rated = ratings[:, 0] == i
        i_ratings = ratings[i_rated, 2]
        i_items = ratings[i_rated, 1]

        i_liked = i_items[i_ratings >= 0]

        # get top N item recommendations for user_i
        i_rec = model.recommend(i)
        i_top_rec = i_rec[:N]

        # check overlap between recommended and liked
        num_liked = np.sum(i_ratings >= 0)
        num_hits = np.in1d(i_top_rec, list(i_liked)).sum()

        liked.append(num_liked)
        hits.append(num_hits)

    precision = np.sum(hits) / (N * len(ratings))  # num_hits / num_recs
    recall = np.sum(hits) / np.sum(liked)  # num_hits / num_liked
    return precision, recall