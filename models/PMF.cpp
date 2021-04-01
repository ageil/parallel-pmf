#include "PMF.h"
#include "ratingsdata.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <thread>

#include <gsl/gsl_assert>

namespace Model
{
    using namespace RatingsData;

    PMF::PMF(const shared_ptr<MatrixXd> &data,
             const int k,
             const double std_beta,
             const double std_theta,
             const vector<int> &users,
             const vector<int> &items)
        : m_data(data), m_k(k), m_std_beta(std_beta), m_std_theta(std_theta), m_users(users), m_items(items)
    {
        cout
            << "Initializing PMF with `data` size " << m_data->rows()
            << " x " << m_data->cols() << " with k=" << k
            << " std_beta=" << std_beta << " std_theta=" << std_theta
            << endl;

        normal_distribution<double> dist_beta(0, std_beta);
        normal_distribution<double> dist_theta(0, std_theta);

        initVectors(dist_theta, m_users, m_theta);
        cout << "Initialized " << m_theta.size() << " users ibtheta map \n";

        initVectors(dist_beta, m_items, m_beta);
        cout << "Initialized " << m_beta.size() << " items in beta map \n";
    }

    PMF::~PMF()
    {
        m_fit_in_progress = false;
    }

    // Initializes map vmap for each entity with random vector of size m_k
    // sampling from distribution dist
    void PMF::initVectors(normal_distribution<> &dist, const vector<int> &entities,
                          map<int, VectorXd> &vmap)
    {
        auto rand = [&]() { return dist(d_generator); };
        for (int elem : entities)
        {
            VectorXd vec = VectorXd::NullaryExpr(m_k, rand);
            vec.normalize();
            vmap[elem] = vec;
        }
    }

    // Evaluate log normal PDF at vector x
    double PMF::logNormPDF(const VectorXd &x, double loc, double scale)
    {
        VectorXd vloc = VectorXd::Constant(x.size(), loc);
        double norm = (x - vloc).norm();
        double log_prob = -log(scale);
        log_prob -= 1.0 / 2.0 * log(2.0 * M_PI);
        log_prob -= 1.0 / 2.0 * (pow(2, norm) / pow(2, scale));
        return log_prob;
    }

    // Evaluate log normal PDF at double x
    double PMF::logNormPDF(double x, double loc, double scale)
    {
        double diff = x - loc;
        double log_prob = -log(scale);
        log_prob -= 1.0 / 2.0 * log(2.0 * M_PI);
        log_prob -= 1.0 / 2.0 * (pow(2.0, diff) / pow(2.0, scale));
        return log_prob;
    }

    // Calculate the log probability of the data under the current model
    void PMF::loss()
    {
        m_fit_in_progress = true;

        while (m_fit_in_progress || !m_loss_queue.empty())
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return !m_loss_queue.empty(); });

            const ThetaBetaSnapshot snapshot = m_loss_queue.front();
            m_loss_queue.pop();

            lock.unlock();

            const auto theta_snapshot = snapshot.theta;
            const auto beta_snapshot = snapshot.beta;

            double loss = 0;

            for (const auto user_id : m_users)
            {
                loss += logNormPDF(theta_snapshot.at(user_id));
            }

            for (const auto item_id : m_items)
            {
                loss += logNormPDF(beta_snapshot.at(item_id));
            }

            for (int idx = 0; idx < m_data->rows(); idx++)
            {
                int i = (*m_data)(idx, 0);
                int j = (*m_data)(idx, 1);

                double r = (*m_data)(idx, 2);
                double r_hat = theta_snapshot.at(i).dot(beta_snapshot.at(j));

                loss += logNormPDF(r, r_hat);
            }

            m_losses.push_back(loss);
            cout << "loss: " << loss << endl;
        }
        cout << "TOTAL COUNT: " << count << endl;
    }

    // Subset m_data by rows where values in column is equal to ID
    MatrixXd PMF::subsetByID(const Ref<MatrixXd> &batch, int ID, int column)
    {
        VectorXi isID = (batch.col(column).array() == ID).cast<int>(); // which rows have ID in col?
        int num_rows = isID.sum();
        int num_cols = batch.cols();
        MatrixXd submatrix(num_rows, num_cols);
        int cur_row = 0;
        for (int i = 0; i < batch.rows(); ++i)
        {
            if (isID[i])
            {
                submatrix.row(cur_row) = batch.row(i);
                cur_row++;
            }
        }
        return submatrix;
    }

    // Fit user preference vectors to sample data in batch m_data[start_row:end_row]
    void PMF::fitUsers(const Ref<MatrixXd> &batch, const double learning_rate)
    {
        MatrixXd users = batch.col(col_value(Cols::user));
        set<int> unique_users = {users.data(), users.data() + users.size()};

        for (auto &usrID : unique_users)
        {
            // extract sub-matrix of user usrID's in batch
            const MatrixXd user_data = subsetByID(batch, usrID, col_value(Cols::user));
            const VectorXi &items = user_data.col(col_value(Cols::item)).cast<int>();
            const VectorXd &ratings = user_data.col(col_value(Cols::rating));

            // compute gradient update of user preference vectors
            VectorXd grad = -(1.0 / m_std_theta) * m_theta[usrID];

            for (int idx = 0; idx < items.size(); idx++)
            {
                int itmID = items(idx);
                double rating = ratings(idx);
                double rating_hat = m_theta[usrID].dot(m_beta[itmID]);
                grad += (rating - rating_hat) * m_beta[itmID];
            }

            VectorXd update = m_theta[usrID] + learning_rate * grad;
            update.normalize();
            m_theta[usrID] = update; // note: no lock needed
        }
    }

    void PMF::fitItems(const Ref<MatrixXd> &batch, const double learning_rate)
    {
        MatrixXd items = batch.col(col_value(Cols::item));
        set<int> unique_items = {items.data(), items.data() + items.size()};

        for (auto &itmID : unique_items)
        {
            // extract sub-matrix of item itmID's data
            const MatrixXd item_data = subsetByID(batch, itmID, col_value(Cols::item));
            const VectorXi &users = item_data.col(col_value(Cols::user)).cast<int>();
            const VectorXd &ratings = item_data.col(col_value(Cols::rating));

            // compute gradient update of item attribute vectors
            VectorXd grad = -(1.0 / m_std_beta) * m_beta[itmID];
            for (int idx = 0; idx < users.size(); idx++)
            {
                int usrID = users(idx);
                double rating = ratings(idx);
                double rating_hat = m_theta[usrID].dot(m_beta[itmID]);
                grad += (rating - rating_hat) * m_theta[usrID];
            }

            VectorXd update = m_beta[itmID] + learning_rate * grad;
            update.normalize();
            m_beta[itmID] = update; // note: no lock needed
        }
    }

    vector<double> PMF::fit(const int epochs, const double gamma, const int batch_size, const int num_threads)
    {
        cout << epochs << " epochs \n"
             << "num_threads: " << num_threads << endl;

        const int max_rows = m_data->rows();
        const int num_batches = max_rows / batch_size;

        Utils::guarded_thread compute_loss([this] { this->loss(); });

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            if (epoch % 10 == 0)
            {
                {
                    lock_guard<mutex> lock(m_mutex);
                    m_loss_queue.emplace(m_theta, m_beta);
                }

                m_cv.notify_one();
                cout << "epoch: " << epoch << endl;
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
                Ref<MatrixXd> batch = m_data->block(row_start, col_start, num_rows, num_cols);

                // add batch fit tasks to thread pool
                threadpool.emplace_back([this, batch, gamma] {
                    this->fitUsers(batch, gamma);
                    this->fitItems(batch, gamma);
                });

                cur_batch += 1;
            }

        } // epochs

        m_fit_in_progress = false;
        return m_losses;
    }

    // Predict ratings using learnebtheta and beta vectors in model
    // Input: data matrix with n rows and 2 columns (user, item)
    // Returns a vector of predicted ratings for each user and item
    VectorXd PMF::predict(const MatrixXd &data)
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

    VectorXd PMF::recommend(int user)
    {
        cerr << "Not implemented yet" << endl;
        VectorXd v;
        return v;
    }

} // namespace Model