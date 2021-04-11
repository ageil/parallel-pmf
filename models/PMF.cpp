#include "PMF.h"
#include "datamanager.h"
#include "utils.h"

#include <iostream>
#include <set>
#include <thread>

#include <gsl/gsl_assert>

namespace Model
{

using namespace Utils;

PMF::PMF(const shared_ptr<DataManager> &data_mgr, const int k, const double std_beta, const double std_theta,
         const int loss_interval)
    : m_data_mgr(data_mgr)
    , m_std_beta(std_beta)
    , m_std_theta(std_theta)
    , m_k(k)
    , m_fit_in_progress(false)
    , m_loss_interval(loss_interval)
{
    Expects(k > 0);
    cout << "[PMF] Initializing PMF with k=" << k << " std_beta=" << m_std_beta << " std_theta=" << m_std_theta << endl;

    const shared_ptr<MatrixXd> training_data = m_data_mgr->getTrain();
    cout << "[PMF] Initializing for fit. Using training data matrix with size: " << training_data->rows() << " x "
         << training_data->cols() << endl;

    normal_distribution<double> dist_beta(0, m_std_beta);
    normal_distribution<double> dist_theta(0, m_std_theta);

    initVectors(dist_theta, *(m_data_mgr->getUsers()), m_theta);
    cout << "[PMF] Initialized " << m_theta.size() << " users in theta map. \n";

    initVectors(dist_beta, *(m_data_mgr->getItems()), m_beta);
    cout << "[PMF] Initialized " << m_beta.size() << " items in beta map. \n";

    cout << "[PMF] Model ready for fitting. \n\n";
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
 */
void PMF::initVectors(normal_distribution<> &dist, const vector<int> &entities, LatentVectors &vmap)
{
    auto rand = [&]() { return dist(d_generator); };
    for (const auto elem : entities)
    {
        VectorXd vec = VectorXd::NullaryExpr(m_k, rand);
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
    Expects(scale > 0.0);

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
    Expects(scale > 0.0);

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
MatrixXd PMF::subsetByID(const Ref<MatrixXd> &batch, const int ID, int column) const
{
    Expects(ID >= 0);
    Expects(column == col_value(Cols::user) || column == col_value(Cols::item));

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
void PMF::computeLoss(const LatentVectors &theta, const LatentVectors &beta)
{

    double loss = 0;
    const vector<int> &user_ids = *(m_data_mgr->getUsers());

    for (const auto user_id : user_ids)
    {
        loss += logNormPDF(theta.at(user_id));
    }

    const vector<int> &item_ids = *(m_data_mgr->getItems());

    for (const auto item_id : item_ids)
    {
        loss += logNormPDF(beta.at(item_id));
    }

    const int user_col = col_value(Cols::user);
    const int item_col = col_value(Cols::item);
    const int rating_col = col_value(Cols::rating);

    const auto &dataMatrix = *(m_data_mgr->getTrain());
    for (int idx = 0; idx < dataMatrix.rows(); idx++)
    {
        int i = dataMatrix(idx, user_col);
        int j = dataMatrix(idx, item_col);

        double r = dataMatrix(idx, rating_col);
        double r_hat = theta.at(i).dot(beta.at(j));

        loss += logNormPDF(r, r_hat);
    }

    m_losses.push_back(loss);
    cout << "[computeLoss] Loss: " << loss << endl;
}

/**
 * Compute the log-likelihood of the snapshots of data found in m_loss_queue (assuming only Gaussian distributions).
 * This queue will wait until it gets a signal that there is a new item to process or until it gets a signal to
 * terminate. If it gets the signal to terminate, it will process any remaining items in the queue before exiting.
 */
void PMF::computeLossFromQueue()
{
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

        const LatentVectorsSnapshot snapshot = [this] {
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
}

/**
 * Compute gradient updates of each user in a batch of data, and apply the update to the corresponding theta vectors.
 * @param batch Reference to a batch of training data containing columns for user IDs, item IDs, and ratings (in order)
 * @param gamma Learning rate to be used in the gradient ascent procedure
 */
void PMF::fitUsers(const Ref<MatrixXd> &batch, const double gamma)
{

    Expects(gamma > 0.0);

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

        VectorXd update = m_theta[usr_id] + gamma * grad;
        update.normalize();
        m_theta[usr_id] = update; // note: no lock needed
    }
}

/**
 * Compute gradient updates of each item in a batch of data, and apply the update to the corresponding beta vectors.
 * @param batch Reference to a batch of training data containing columns for user IDs, item IDs, and ratings (in order)
 * @param gamma Learning rate to be used in the gradient ascent procedure
 */
void PMF::fitItems(const Ref<MatrixXd> &batch, const double gamma)
{

    Expects(gamma > 0.0);

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

        VectorXd update = m_beta[itm_id] + gamma * grad;
        update.normalize();
        m_beta[itm_id] = update; // note: no lock needed
    }
}

/**
 * Fit the latent beta and theta vectors to the training dataset sequentially. This performs the loss computation every
 * 10 epochs sequentially.
 * @param epochs Number of times the training dataset is passed over in order to compute gradient updates
 * @param gamma Learning rate to be used in the gradient ascent procedure
 * @return A vector of log-likelihoods of the data under the model for each epoch
 */
vector<double> PMF::fitSequential(const int epochs, const double gamma)
{
    Expects(epochs > 0);
    Expects(gamma > 0.0);

    cout << "[fitSequential] Running fit (sequential) on main thread. Computing loss every " << m_loss_interval
         << " epochs.\n\n";

    const shared_ptr<MatrixXd> data_matrix_ptr = m_data_mgr->getTrain();

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        if (epoch % m_loss_interval == 0)
        {
            computeLoss(m_theta, m_beta);
            cout << "[fitSequential] Epoch: " << epoch << endl;
        }

        fitUsers(*data_matrix_ptr, gamma);
        fitItems(*data_matrix_ptr, gamma);

    } // epochs

    return m_losses;
}

/**
 * Fit the latent beta and theta vectors to the training dataset in parallel over multiple threads.This performs the
 * loss computation every 10 epochs in parallel on a separate thread.
 * @param epochs Number of times the training dataset is passed over in order to compute gradient updates
 * @param gamma Learning rate to be used in the gradient ascent procedure
 * @param n_threads Number of threads the training dataset to distribute the dataset over
 * @return A vector of log-likelihoods of the data under the model for each epoch
 */
vector<double> PMF::fitParallel(const int epochs, const double gamma, const int n_threads)
{
    Expects(epochs > 0);
    Expects(gamma > 0.0);
    Expects(n_threads > 0);

    const auto train_data_ptr = m_data_mgr->getTrain();
    const int max_rows = train_data_ptr->rows();
    int batch_size = max_rows / (n_threads - 1); // (n-1) threads for params. update, 1 thread for loss calculation
    const int num_batches = max_rows / batch_size;

    cout << "[fitParallel] Using " << n_threads << " threads" << endl
         << "[fitParallel] Total epochs: " << epochs << endl
         << "[fitParallel] max rows: " << max_rows << endl
         << "[fitParallel] batch size: " << batch_size << endl
         << "[fitParallel] num batches: " << num_batches << endl
         << "[fitParallel] Computing loss every " << m_loss_interval << " epochs\n\n";

    Utils::guarded_thread compute_loss_thread([this] {
        cout << "[computeLossThread] Loss computation thread started.\n";
        this->computeLossFromQueue();
        cout << "[computeLossThread] Loss computation thread completed.\n\n";
    });

    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        if (epoch % m_loss_interval == 0)
        {
            {
                lock_guard<mutex> lock(m_mutex);
                m_loss_queue.emplace(m_theta, m_beta);
            }

            m_cv.notify_one();
            cout << "[fitParallel] Epoch: " << epoch << endl;
        }

        vector<Utils::guarded_thread> threads;

        int cur_batch = 0;
        while (cur_batch <= num_batches)
        {
            // compute start/end indices for current batch
            const int row_start = cur_batch * batch_size;
            const int num_rows = min(max_rows - row_start, batch_size);
            const int col_start = col_value(Cols::user);
            const int num_cols = col_value(Cols::rating) + 1;

            // reference batch of data
            Ref<MatrixXd> batch = train_data_ptr->block(row_start, col_start, num_rows, num_cols);

            // add batch fit tasks to thread pool
            threads.emplace_back([this, batch, gamma] {
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

LatentVectors &PMF::getTheta()
{
    return m_theta;
}

LatentVectors &PMF::getBeta()
{
    return m_beta;
}

vector<double> &PMF::getComputedLoss()
{
    return m_losses;
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
 * @return A list of recommended item IDs sorted from most to least recommended
 */
VectorXi PMF::recommend(const int user_id, const int N) const
{
    Expects(N >= 1);
    Expects(m_theta.count(user_id) > 0);

    vector<double> vi_items{};
    for (auto it : m_beta)
    {
        vi_items.push_back(it.first);
    }

    Eigen::Map<VectorXd> items(vi_items.data(), vi_items.size());
    VectorXd user(items.size());
    user.setConstant(user_id);

    const int user_col = col_value(Cols::user);
    const int item_col = col_value(Cols::item);

    MatrixXd usr_data(items.size(), 2);
    usr_data.col(user_col) = user;
    usr_data.col(item_col) = items;

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

/**
 * Generate a vector of top N most recommended items with actual item_names for user with ID user_id.
 * @param user_id User ID of the user to generate item recommendations
 * @param item_name Hashmap of of item ID (int) to their item item_name (string)
 * @param N Number of item recommendations to generate
 * @return A list of recommended items names sorted from most to least recommended
 */
vector<string> PMF::recommend(const int user_id, const unordered_map<int, string> &item_name, const int N) const
{
    Expects(N >= 1);
    Expects(m_theta.count(user_id) > 0);

    // Get top N item recommendations for user
    VectorXi rec = recommend(user_id, N);
    vector<string> rec_names;

    for (int i = 0; i < rec.size(); i++)
    {
        rec_names.push_back(item_name.at(rec[i]));
    }

    return rec_names;
}

/**
 * Generate a vector of top N most similar items to the input item with Item ID
 * @param item_id Item ID of the item to generate item recommendations
 * @param id_name Map of of item ID (int) to their item item_name (string)
 * @param N Number of item recommendations to generate
 * @return A list of recommended items names sorted from the most to least similar to the input item
 */
vector<string> PMF::getSimilarItems(int &item_id, unordered_map<int, string> &id_name, const int N)
{
    Expects(N > 0);
    Expects(m_beta.count(item_id) > 0);

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

} // namespace Model
