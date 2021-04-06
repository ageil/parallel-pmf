#include "PMF.h"
#include "../csvlib/csv.h"
#include "datamanager.h"
#include "ratingsdata.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>

#include <gsl/gsl_assert>

namespace Model
{

using namespace Utils;
using namespace RatingsData;

PMF::PMF(const shared_ptr<DataManager::DataManager> &data_mgr, const int k, const double std_beta,
         const double std_theta)
    : m_data_mgr(data_mgr)
    , m_training_data(data_mgr->getTrain())
    , m_std_beta(std_beta)
    , m_std_theta(std_theta)
    , m_fit_in_progress(false)
{
    cout << "[PMF] Initializing PMF with `data` size " << m_training_data->rows() << " x " << m_training_data->cols()
         << " with k=" << k << " std_beta=" << std_beta << " std_theta=" << std_theta << endl;

    normal_distribution<double> dist_beta(0, std_beta);
    normal_distribution<double> dist_theta(0, std_theta);

    initVectors(dist_theta, m_data_mgr->users, m_theta, k);
    cout << "[PMF] Initialized " << m_theta.size() << " users in theta map \n";

    initVectors(dist_beta, m_data_mgr->items, m_beta, k);
    cout << "[PMF] Initialized " << m_beta.size() << " items in beta map \n";
    cout << "[PMF] Initialization complete. \n\n";
}

PMF::~PMF()
{
    m_fit_in_progress = false;
}

/**
 * Initialize for each entity the corresponding k-length latent vector in vmap by drawing randomly from dist.
 * @param dist The distribution from which entry values for the latent vector are randomly drawn
 * @param entities A vector of entity IDs, either user IDs or item IDs
 * @param vmap A map connecting each entity ID to its corresponding latent vector
 * @param k The length of each latent vector
 */
void PMF::initVectors(normal_distribution<> &dist, const vector<int> &entities, map<int, VectorXd> &vmap, const int k)
{
    auto rand = [&]() { return dist(d_generator); };
    for (int elem : entities)
    {
        VectorXd vec = VectorXd::NullaryExpr(k, rand);
        vec.normalize();
        vmap[elem] = vec;
    }
}

/**
 * Compute the log-likelihood of a vector x under a Gaussian distribution with mean loc and standard deviation scale.
 * @param x A vector of doubles to be evaluated
 * @param loc The mean of the Gaussian distribution
 * @param scale The standard deviation of the Gaussian distribution
 * @return The log-probability of observing x
 */
double PMF::logNormPDF(const VectorXd &x, double loc, double scale) const
{
    VectorXd vloc = VectorXd::Constant(x.size(), loc);
    double norm = (x - vloc).norm();
    double log_prob = -log(scale);
    log_prob -= 1.0 / 2.0 * log(2.0 * M_PI);
    log_prob -= 1.0 / 2.0 * (pow(2, norm) / pow(2, scale));
    return log_prob;
}

/**
 * Compute the log-likelihood of a double x under a Gaussian distribution with mean loc and standard deviation scale.
 * @param x A point double to be evaluated
 * @param loc The mean of the Gaussian distribution
 * @param scale The standard deviation of the Gaussian distribution
 * @return The log-probability of observing x
 */
double PMF::logNormPDF(double x, double loc, double scale) const
{
    double diff = x - loc;
    double log_prob = -log(scale);
    log_prob -= 1.0 / 2.0 * log(2.0 * M_PI);
    log_prob -= 1.0 / 2.0 * (pow(2.0, diff) / pow(2.0, scale));
    return log_prob;
}

/**
 * Extract a subset of a data batch where the value in column is ID.
 * @param batch Reference to a batch of data
 * @param ID The ID of a user or item to be extracted
 * @param column Index of either the user or item column in which ID is located
 * @return A matrix of rows where values in column are all ID
 */
MatrixXd PMF::subsetByID(const Ref<MatrixXd> &batch, int ID, int column) const
{
    VectorXi is_id = (batch.col(column).array() == ID).cast<int>(); // which rows have ID in col?
    int num_rows = is_id.sum();
    int num_cols = batch.cols();
    MatrixXd submatrix(num_rows, num_cols);
    int cur_row = 0;
    for (int i = 0; i < batch.rows(); ++i)
    {
        if (is_id[i])
        {
            submatrix.row(cur_row) = batch.row(i);
            cur_row++;
        }
    }
    return submatrix;
}

/**
 * Compute the log-likelihood of the data under the model (assuming only Gaussian distributions).
 * @param theta Map of user IDs to user preference vectors
 * @param beta Map of item IDs to item attribute vectors
 */
void PMF::computeLoss(const map<int, VectorXd> &theta, const map<int, VectorXd> &beta)
{
    double loss = 0;

    for (const auto user_id : m_data_mgr->users)
    {
        loss += logNormPDF(theta.at(user_id));
    }

    for (const auto item_id : m_data_mgr->items)
    {
        loss += logNormPDF(beta.at(item_id));
    }

    const auto &dataMatrix = *m_training_data;
    for (int idx = 0; idx < dataMatrix.rows(); idx++)
    {
        int i = dataMatrix(idx, 0);
        int j = dataMatrix(idx, 1);

        double r = dataMatrix(idx, 2);
        double r_hat = theta.at(i).dot(beta.at(j));

        loss += logNormPDF(r, r_hat);
    }

    m_losses.push_back(loss);
    cout << "[computeLoss] Loss: " << loss << endl;
}

void PMF::computeLossFromQueue()
{
    cout << "[computeLossFromQueue] Loss computation thread started.\n";
    m_fit_in_progress = true;

    while (m_fit_in_progress || !m_loss_queue.empty())
    {
        {
            // Waits for the signal that there are items on the m_loss_queue
            // to process or the signal to terminate the thread.
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return !(m_fit_in_progress && m_loss_queue.empty()); });
        }

        if (!m_fit_in_progress && m_loss_queue.empty())
        {
            return;
        }

        Expects(!m_loss_queue.empty());

        const ThetaBetaSnapshot snapshot = [this] {
            const auto snapshot_tmp = m_loss_queue.front();
            {
                lock_guard<mutex> lock(m_mutex);
                m_loss_queue.pop();
            }
            return snapshot_tmp;
        }();

        const auto theta_snapshot = snapshot.theta;
        const auto beta_snapshot = snapshot.beta;

        computeLoss(theta_snapshot, beta_snapshot);
    }

    cout << "[computeLossFromQueue] Loss computation thread completed.\n\n";
}

/**
 * Compute gradient updates of each user in a batch of data, and apply the update to the corresponding theta vectors.
 * @param batch Reference to a batch of training data containing columns for user IDs, item IDs, and ratings (in order)
 * @param learning_rate Learning rate to be used in the gradient ascent procedure
 */
void PMF::fitUsers(const Ref<MatrixXd> &batch, const double learning_rate)
{
    MatrixXd users = batch.col(col_value(Cols::user));
    set<int> unique_users = {users.data(), users.data() + users.size()};

    for (const auto usr_id : unique_users)
    {
        // extract sub-matrix of user usrID's in batch
        const MatrixXd user_data = subsetByID(batch, usr_id, col_value(Cols::user));
        const VectorXi &items = user_data.col(col_value(Cols::item)).cast<int>();
        const VectorXd &ratings = user_data.col(col_value(Cols::rating));

        // compute gradient update of user preference vectors
        VectorXd grad = -(1.0 / m_std_theta) * m_theta[usr_id];

        for (int idx = 0; idx < items.size(); idx++)
        {
            int itmID = items(idx);
            double rating = ratings(idx);
            double rating_hat = m_theta[usr_id].dot(m_beta[itmID]);
            grad += (rating - rating_hat) * m_beta[itmID];
        }

        VectorXd update = m_theta[usr_id] + learning_rate * grad;
        update.normalize();
        m_theta[usr_id] = update; // note: no lock needed
    }
}

/**
 * Compute gradient updates of each item in a batch of data, and apply the update to the corresponding beta vectors.
 * @param batch Reference to a batch of training data containing columns for user IDs, item IDs, and ratings (in order)
 * @param learning_rate Learning rate to be used in the gradient ascent procedure
 */
void PMF::fitItems(const Ref<MatrixXd> &batch, const double learning_rate)
{
    MatrixXd items = batch.col(col_value(Cols::item));
    set<int> unique_items = {items.data(), items.data() + items.size()};

    for (const auto itm_id : unique_items)
    {
        // extract sub-matrix of item itmID's data
        const MatrixXd item_data = subsetByID(batch, itm_id, col_value(Cols::item));
        const VectorXi &users = item_data.col(col_value(Cols::user)).cast<int>();
        const VectorXd &ratings = item_data.col(col_value(Cols::rating));

        // compute gradient update of item attribute vectors
        VectorXd grad = -(1.0 / m_std_beta) * m_beta[itm_id];
        for (int idx = 0; idx < users.size(); idx++)
        {
            int usrID = users(idx);
            double rating = ratings(idx);
            double rating_hat = m_theta[usrID].dot(m_beta[itm_id]);
            grad += (rating - rating_hat) * m_theta[usrID];
        }

        VectorXd update = m_beta[itm_id] + learning_rate * grad;
        update.normalize();
        m_beta[itm_id] = update; // note: no lock needed
    }
}

/**
 * Fit the latent beta and theta vectors to the training dataset sequentially.
 * @param epochs Number of times the training dataset is passed over in order to compute gradient updates
 * @param gamma Learning rate to be used in the gradient ascent procedure
 * @return A vector of log-likelihoods of the data under the model for each epoch
 */
vector<double> PMF::fitSequential(const int epochs, const double gamma)
{
    cout << "Running fit (sequential) on main thread." << endl << endl;

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        if (epoch % 10 == 0)
        {
            computeLoss(m_theta, m_beta);
            cout << "[fitSequential] Epoch: " << epoch << endl;
        }

        fitUsers(*m_training_data, gamma);
        fitItems(*m_training_data, gamma);

    } // epochs

    return m_losses;
}

/**
 * Fit the latent beta and theta vectors to the training dataset in parallel over multiple threads.
 * @param epochs Number of times the training dataset is passed over in order to compute gradient updates
 * @param gamma Learning rate to be used in the gradient ascent procedure
 * @param n_threads Number of threads the training dataset to distribute the dataset over
 * @return A vector of log-likelihoods of the data under the model for each epoch
 */
vector<double> PMF::fitParallel(const int epochs, const double gamma, const int n_threads)
{
    const int max_rows = m_training_data->rows();
    int batch_size = max_rows / (n_threads - 1); // (n-1) threads for params. update, 1 thread for loss calculation
    const int num_batches = max_rows / batch_size;

    cout << "[fitParallel] Using " << n_threads << " threads" << endl
         << "[fitParallel] Total epochs: " << epochs << endl
         << "[fitParallel] max rows: " << max_rows << endl
         << "[fitParallel] batch size: " << batch_size << endl
         << "[fitParallel] num batches: " << num_batches << endl
         << endl;

    Utils::guarded_thread compute_loss_thread([this] { this->computeLossFromQueue(); });

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        if (epoch % 10 == 0)
        {
            {
                lock_guard<mutex> lock(m_mutex);
                m_loss_queue.emplace(m_theta, m_beta);
            }

            m_cv.notify_one();
            cout << "[fitParallel] Epoch: " << epoch << endl;
        }

        vector<Utils::guarded_thread> threadpool;

        int cur_batch = 0;
        while (cur_batch <= num_batches)
        {
            // compute start/end indices for current batch
            const int row_start = cur_batch * batch_size;
            const int num_rows = min(max_rows - row_start, batch_size);
            const int col_start = col_value(Cols::user);
            const int num_cols = col_value(Cols::rating) + 1;

            // reference batch of data
            Ref<MatrixXd> batch = m_training_data->block(row_start, col_start, num_rows, num_cols);

            // add batch fit tasks to thread pool
            threadpool.emplace_back([this, batch, gamma] {
                this->fitUsers(batch, gamma);
                this->fitItems(batch, gamma);
            });

            cur_batch += 1;
        }

    } // epochs

    m_fit_in_progress = false;
    m_cv.notify_one();

    return m_losses;
}

void PMF::load(filesystem::path &indir)
{
    cout << "Loading previously learnt parameters into model..." << endl;

    filesystem::path theta_fname = indir / "theta.csv";
    filesystem::path beta_fname = indir / "beta.csv";
    if (!filesystem::exists(theta_fname) || !filesystem::exists(beta_fname))
    {
        cerr << "Model doesn't have learnt parameters, need to fit data first" << endl;
        exit(1);
    }

    loadModel(theta_fname, LatentVar::theta);
    loadModel(beta_fname, LatentVar::beta);
}

void PMF::loadModel(filesystem::path &indir, LatentVar option)
{
    io::CSVReader<2> in(indir);
    in.read_header(io::ignore_extra_column, "id", "vector");
    int id;
    string str;

    while (in.read_row(id, str))
    {
        vector<string> tokenized = Utils::tokenize(str);
        vector<double> vi_params(tokenized.size());
        std::transform(tokenized.begin(), tokenized.end(), vi_params.begin(),
                       [&](const string &s) { return std::stod(s); });
        Eigen::Map<VectorXd> params(vi_params.data(), vi_params.size());

        if (option == LatentVar::theta)
        {
            m_theta[id] = params;
        }
        else
        {
            m_beta[id] = params;
        }
    }
}

void PMF::save(filesystem::path &outdir)
{
    if (!filesystem::exists(outdir))
    {
        filesystem::create_directory(outdir);
    }

    cout << "Saving loss values..." << endl;
    filesystem::path loss_fname = outdir / "loss.csv";
    ofstream loss_file;
    loss_file.open(loss_fname);
    loss_file << "Loss";
    for (auto &loss : m_losses)
    {
        loss_file << endl << loss;
    }
    loss_file.close();

    cout << "Saving model parameters..." << endl;
    filesystem::path theta_fname = outdir / "theta.csv";
    ofstream theta_file;
    theta_file.open(theta_fname);
    theta_file << "id,vector";

    for (auto const &[id, theta_i] : m_theta)
    {
        theta_file << endl << id << ',' << theta_i.transpose();
    }
    theta_file.close();

    filesystem::path beta_fname = outdir / "beta.csv";
    ofstream beta_file;
    beta_file.open(beta_fname);
    beta_file << "id,vector";

    for (auto const &[id, beta_i] : m_beta)
    {
        beta_file << endl << id << ',' << beta_i.transpose();
    }
    beta_file.close();
}

/**
 * Predict ratings using learnt theta and beta vectors in model.
 * @param data A 2-column matrix with the first column denoting user IDs and the second column denoting item IDs
 * @return A vector of predicted ratings for each pair of user and item IDs
 */
VectorXd PMF::predict(const MatrixXd &data) const
{
    Expects(data.cols() == 2);
    const int num_rows = data.rows();

    VectorXd predictions(num_rows);
    for (int i = 0; i < num_rows; ++i)
    {
        int user = data(i, col_value(Cols::user));
        int item = data(i, col_value(Cols::item));

        const VectorXd &userTheta = m_theta.at(user);
        const VectorXd &itemBeta = m_beta.at(item);

        predictions(i) = userTheta.dot(itemBeta);
    }

    return predictions;
}

/**
 * Generate a vector of top N most recommended items for user with ID user_id.
 * @param user_id User ID of the user to generate item recommendations
 * @param N Number of item recommendations to generate
 * @return A list of recommended items sorted from most to least recommended
 */
VectorXi PMF::recommend(const int user_id, const int N) const
{
    vector<double> vi_items{};
    for (auto &it : m_beta)
    {
        vi_items.push_back(it.first);
    }

    Eigen::Map<VectorXd> items(vi_items.data(), vi_items.size());
    VectorXd user(items.size());
    user.setConstant(user_id);

    MatrixXd usr_data(items.size(), 2);
    usr_data.col(0) = user;
    usr_data.col(1) = items;

    VectorXd predictions = predict(usr_data);
    VectorXi item_indices = Utils::argsort(predictions, Order::descend);
    VectorXi items_rec(items.size()); // all items for the current user(most
                                      // recommended --> least recommended)
    for (int i = 0; i < items.size(); i++)
    {
        items_rec[i] = items[item_indices[i]];
    }

    // return the top N recommendations for the current user
    VectorXi top_rec = items_rec.topRows(N);

    return top_rec;
}

vector<string> PMF::recommend(const int user_id, const unordered_map<int, string> &item_name, const int N) const
{
    // Get top N item recommendations for user
    VectorXi rec = recommend(user_id, N);
    vector<string> rec_names;

    for (int i = 0; i < rec.size(); i++)
    {
        rec_names.push_back(item_name.at(rec[i]));
    }

    return rec_names;
}

vector<string> PMF::recommendByGenre(string &genre, unordered_map<int, string> &id_name,
                                     unordered_map<string, unordered_set<int>> genre_ids, const int N)
{
    vector<string> similar_items{};
    if (genre_ids.find(genre) == genre_ids.end())
    {
        cerr << "Didn't find the genre " << genre << " in current dataset..." << endl;
        return similar_items;
    }

    // Random pick a movie with the input genre
    unordered_set<int> id_set = genre_ids[genre];
    default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(0, id_set.size() - 1);
    auto itr = id_set.begin();
    int loc = dist(generator);
    std::advance(itr, loc);
    int rand_id = *itr;

    // Return top N items most similar to the selected item
    return getSimilarItems(rand_id, id_name, N);
}

vector<string> PMF::getSimilarItems(int &item_id, unordered_map<int, string> &id_name, const int N)
{
    VectorXd beta_item_id = m_beta.at(item_id);
    vector<double> similarities{};
    unordered_map<double, int> similarity_id{};

    for (auto const &[i, beta_i] : m_beta)
    {
        if (i != item_id)
        {
            double similarity = Utils::cosine(beta_item_id, beta_i);
            similarities.push_back(similarity);
            similarity_id[similarity] = i;
        }
    }

    // Return N most similar items
    vector<string> similar_items{};
    std::sort(similarities.begin(), similarities.end(), std::greater<>());
    for (int i = 0; i < N; i++)
    {
        int id = similarity_id[similarities[i]];
        similar_items.push_back(id_name[id]);
    }

    return similar_items;
}

// Return the precision & recall of the top N predicted items for each user in
// the give dataset
Metrics PMF::accuracy(const shared_ptr<MatrixXd> &data, const int N) const
{
    double num_likes_total = 0;
    double num_hits_total = 0;

    MatrixXd users = (*data).col(col_value(Cols::user));
    set<int> unique_users = {users.data(), users.data() + users.size()};

    for (auto &user_id : unique_users)
    {
        const MatrixXd user_data = subsetByID(*data, user_id, col_value(Cols::user));
        const VectorXi &items = user_data.col(col_value(Cols::item)).cast<int>();
        const VectorXd &ratings = user_data.col(col_value(Cols::rating));

        // Get all items with non-negative ratings ("likes") from the current
        // user id
        vector<int> pos_idxs = Utils::nonNegativeIdxs(ratings);
        VectorXi user_liked(pos_idxs.size());
        for (int i = 0; i < pos_idxs.size(); i++)
        {
            user_liked[i] = static_cast<int>(items[pos_idxs[i]]); // item type: int
        }

        // Get top N recommendations for the current user_id
        VectorXi top_rec = recommend(user_id, N);

        // Get overlap between recommendation & ground-truth "likes"
        int num_likes = pos_idxs.size();
        int num_hits = Utils::countIntersect(top_rec, user_liked);
        num_likes_total += num_likes;
        num_hits_total += num_hits;
    }

    Metrics acc;
    acc.precision = num_hits_total / static_cast<double>(N * unique_users.size());
    acc.recall = num_hits_total / num_likes_total;

    return acc;
}

} // namespace Model
