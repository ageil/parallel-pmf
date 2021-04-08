import os
import subprocess
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from pmf import plot as pmf_plot

class PMF(object):
    """
    Python wrapper for our C++ program of Probablistic Matrix Factorizattion (PMF)
    """
    def __init__(self, indir, outdir, task='train', **kwargs):
        pwd = os.path.join(os.path.dirname(__file__), 'bin')
        exec_name = 'main.tsk'
        self.bin_exec = os.path.join(pwd, exec_name)
        self.indir = indir
        self.data_path = os.path.join(indir, 'ratings.csv')
        self.mapper_path = os.path.join(indir, 'movies.csv')
        self.outdir = outdir
        self.task = task

        # model parameters
        self.theta = pd.DataFrame()  # pd.DataFrame of user_id -> k-dimensional latent attribute
        self.beta = pd.DataFrame() # pd.DataFrame of user_id -> k-dimensional latent attribute
        self.users = set()
        self.items = set()
        self.genres = set()
        self.loss = pd.DataFrame()

        # mapping information betweeen item, item title & genre
        self.item_title = {}
        self.item_genre = {}
        self.title_item = {}
        self.title_genre = {}
        self.genre_items = {}

        assert os.path.exists(self.bin_exec), "Invalid binary executable"
        assert os.path.exists(self.data_path), "Input data file {} doesn't exist".format(self.data_path)
        assert os.path.exists(self.mapper_path), "Input mapping file {} doesn't exist".format(self.mapper_path)

        self._initialize(kwargs)

    def _initialize(self, kwargs):
        self.default_params = {
            'parallel': True,
            'thread': 8,
            'gamma': 0.01,
            'theta_std': 1,
            'beta_std': 1,
        }
        print('Initializing model parameters...')

        self.args = {}
        for key in self.default_params.keys():
            if key in kwargs.keys():
                self.args[key] = kwargs[key]
            else:
                self.args[key] = self.default_params[key]

    def learn(self, k=3, n_epochs=200, train_test_split=0.7):
        cmd = [self.bin_exec,
               "--task {}".format(self.task),
               "-i {}".format(self.data_path),
               "-m {}".format(self.mapper_path),
               "-o {}".format(self.outdir),
               "-k {}".format(k), "-n {}".format(n_epochs),
               "-r {}".format(train_test_split),
               "--thread {}".format(self.args['thread']),
               "--gamma {}".format(self.args['gamma']),
               "--std_theta {}".format(self.args['theta_std']),
               "--std_beta {}".format(self.args['beta_std'])]

        print('Training model...')
        res = subprocess.getoutput(' '.join(cmd))
        print(res)

    def load(self, indir):
        print('Loading previously learnt parameters into model...')
        theta_file = os.path.join(indir, "theta.csv")
        beta_file = os.path.join(indir, "beta.csv")
        loss_file = os.path.join(indir, "loss.csv")

        assert os.path.exists(loss_file), \
            "Loglikelihood hasn't been calculated, please train the model first"

        assert os.path.exists(theta_file) and os.path.exists(beta_file), \
            "Latent vector theta & beta hasn't been learnt, please train the model first"

        self.loss = pd.read_csv(loss_file)
        self.theta = self._load_model(theta_file)
        self.beta = self._load_model(beta_file)
        self.users = set(self.theta.index)
        self.items = set(self.beta.index)
        self._load_mapper()  # Load item - title - genre maps

    def _verify_load_status(self):
        if len(self.theta) == 0 or len(self.beta) == 0:
            self.load(self.outdir)

    def _load_model(self, file):
        df = pd.read_csv(file)
        return self._process_vectors(df)

    def _load_mapper(self):
        assert os.path.exists(self.mapper_path), \
            "Item mapping file doesn't exist"
        df_mapper = pd.read_csv(self.mapper_path)

        first_genre = df_mapper['genres'].apply(lambda x: x.strip().split('|')[0])
        df_mapper['first_genre'] = first_genre
        self.genres = set(np.unique(first_genre))
        df_reidx1 = df_mapper.set_index('movieId')
        df_reidx2 = df_mapper.set_index('title')

        self.item_title = df_reidx1.to_dict()['title']
        self.item_genre = df_reidx1.to_dict()['genres']
        item_first_genre = df_reidx1.to_dict()['first_genre']
        self.title_item = df_reidx2.to_dict()['movieId']
        self.title_genre = df_reidx2.to_dict()['genres']

        for item, genre in item_first_genre.items():
            if genre in self.genre_items.keys():
                self.genre_items[genre].add(item)
            else:
                self.genre_items[genre] = {item}

    def _process_vectors(self, df):
        df['vector'] = df['vector'].apply(lambda x: [float(i) for i in x.split()])
        vals = np.array(df['vector'].values.tolist())
        cols = ['attr_' + str(i) for i in range(1, vals.shape[1] + 1)]
        df_processed = pd.DataFrame(data=vals, index=df['id'], columns=cols)

        return df_processed

    def _predict(self, user_id):
        """Predict the preference of user_id to all items"""
        theta_i = self.theta.loc[user_id]
        pred = theta_i.dot(self.beta.T)

        return pred

    def _predict_user(self, item_id):
        beta_i = self.beta.loc[item_id]
        pred = beta_i.dot(self.theta.T)

        return pred

    def _recommend_item_to_user(self, item_id, N=10):
        self._verify_load_status()
        preds = self._predict_user(item_id)
        rec_users = preds.sort_values(ascending=False)
        rec_use_ids = rec_users.index[:N].to_series()

        return rec_use_ids

    def recommend_user(self, user_id, N=10, verbose=1):
        self._verify_load_status()
        assert user_id in self.users, \
            "User id {} doesn't exist in the dataset".format(user_id)
        if verbose:
            print("Top {0} recommended movies for user {1}:".format(N, user_id))

        preds = self._predict(user_id)
        rec_items = preds.sort_values(ascending=False)
        rec_items = rec_items.index[:N].to_series()
        rec_titles = rec_items.map(self.item_title)
        df_rec = self._refactor_rec(rec_titles)

        return df_rec

    def recommend_items(self, item, N=10, verbose=1):
        self._verify_load_status()
        if isinstance(item, str):
            try:
                item_id = self.title_item[item]
            except KeyError:
                print("Item {} doesn't exist in the dataset".format(item))
        else:
            item_id = item
        assert item_id in self.items, \
            "Item id {0} doesn't exist in the dataset".format(item_id)
        if verbose:
             print("Top {0} recommended movies if you also like {1}:".format(N, self.item_title[item_id]))

        rec_items = self._get_similar_items(item_id, N)
        rec_titles = rec_items.map(self.item_title)
        df_rec = self._refactor_rec(rec_titles)

        return df_rec

    def recommend_joint(self, user_id, iter=2, N=3, verbose=1):
        assert user_id in self.users, \
            "User {} doesn't exist in the dataset".format(user_id)
        if verbose:
            print("Iteratively recommending users and items for {} periods...".format(iter))

        rec_users = {user_id}
        rec_items = set()
        curr_itr = 0

        while curr_itr < iter:
            for user in rec_users:  # recommend user -> item
                curr_items = self.recommend_user(user, N=N, verbose=0).index
                rec_items = rec_items.union(set(curr_items))

            for item in rec_items:  # recommend item -> user
                curr_users = self._recommend_item_to_user(item, N=N)
                rec_users = rec_users.union(set(curr_users))

            curr_itr += 1

        rec_users = list(rec_users)
        rec_items = pd.Series(list(rec_items))
        rec_titles = rec_items.map(self.item_title)
        df_rec_items = self._refactor_rec(rec_titles, idx=rec_items)

        return rec_users, df_rec_items

    def recommend_genre(self, genre, N=10, verbose=1):
        self._verify_load_status()
        assert genre in self.genres, \
            "Genre {} doesen't exist in the dataset".format(genre)
        if verbose:
            print("Top {0} recommended movies for genre {1}:".format(N, genre))

        candidate_item_list = list(self.genre_items[genre])
        sample_item_id = np.random.choice(candidate_item_list)
        rec_items = self._get_similar_items(sample_item_id, N)
        rec_titles = rec_items.map(self.item_title)
        df_rec = self._refactor_rec(rec_titles)

        return df_rec

    def _get_similar_items(self, item_id, N):
        beta_j = self.beta.loc[item_id]
        similarity = beta_j.dot(self.beta.T)
        items = similarity.sort_values(ascending=False).index[1:N+1].to_series() # skip index[0] to avoid self
        return items

    def _refactor_rec(self, rec, idx=None):
        genres = rec.map(self.title_genre)
        idx = rec.index if idx is None else idx
        df_rec = pd.DataFrame(zip(rec, genres), index=idx, columns=['title', 'genre'])

        return df_rec

    def display_loss(self, display=True, save=True):
        self._verify_load_status()
        x = np.arange(self.loss.shape[0]) * 10 + 10
        self.loss['Epoch'] = x
        pmf_plot.loss(self.loss, outdir=self.outdir)

    def display_user(self, user_id, N=10, show_title=False, interactive=True):
        self._verify_load_status()
        print("Spatial visualization of top {0} recommended movies for user {1}...".format(N, user_id))

        df_rec = self.recommend_user(user_id, N)
        print(df_rec.head())

        vec = self.beta.loc[df_rec.index].values
        titles = df_rec['title']

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title, is_similar=True)
        else:
            pmf_plot.arrow(vec)

    def display_item(self, item, N=10, show_title=False, interactive=True):
        self._verify_load_status()
        title = item if isinstance(item, str) else self.item_title[item]
        print("Spatial visualization of top {0} similar movies for item {1}...".format(N, title))

        df_rec = self.recommend_items(item, N)
        print(df_rec.head())

        vec = self.beta.loc[df_rec.index].values
        titles = df_rec['title']

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title, is_similar=True)
        else:
            pmf_plot.arrow(vec)

    def display_genre(self, genre, N=10, show_title=False, interactive=True):
        self._verify_load_status()
        assert genre in self.genres, \
            "Genre {} doesen't exist in the dataset".format(genre)

        rand_ids = np.random.choice(list(self.genre_items[genre]), N)
        vec = self.beta.loc[rand_ids].values
        titles = pd.Series(rand_ids).map(self.item_title)

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title, is_similar=True)
        else:
            pmf_plot.arrow(vec)

    def display_joint(self, user_id, iter=2, N=10, show_title=False, interactive=True):
        """Iteratively plot interacting users & items"""
        self._verify_load_status()
        user_ids, df_items = self.recommend_joint(user_id, iter=iter)

        vec_users = self.theta.loc[user_ids].values
        vec_items = self.beta.loc[df_items.index].values
        titles = df_items['title']

        if interactive:
            pmf_plot.arrow_joint_interactive(vec_users, vec_items, titles, show_title=show_title)
        else:
            pmf_plot.arrow_joint(vec_users, vec_items)

    def display_random(self, N=3, n_neighbors=10, show_title=False, interactive=True):
        self._verify_load_status()
        print('Spatial visualization of the neighbors of {} random items'.format(N))

        rand_ids = np.random.choice(list(self.items), N)
        indices = set()
        for id in rand_ids:
            idx = self.recommend_items(id, n_neighbors, verbose=0).index
            indices = indices.union(idx)

        vec = self.beta.loc[indices].values
        titles = pd.Series(list(indices)).map(self.item_title)

        if interactive:
            pmf_plot.arrow_interactive(vec, titles, show_title=show_title)
        else:
            pmf_plot.arrow(vec)
