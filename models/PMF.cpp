#include "PMF.h"
#include "ratingsdata.h"
#include <cmath>
#include <set>
#include <thread>
#include <gsl/gsl_assert>

namespace Model
{
    using namespace RatingsData;

    PMF::PMF(const shared_ptr<MatrixXd> &data, const int k, const double std_beta, const double std_theta)
        : m_data(data), m_k(k), m_std_beta(std_beta), m_std_theta(std_theta), m_run_compute_loss(false)
    {
        cout
            << "Initializing PMF with `data` size " << m_data->rows()
            << " x " << m_data->cols() << " with k=" << k
            << " std_beta=" << std_beta << " std_theta=" << std_theta
            << endl;

        m_users = getUnique(col_value(Cols::user));
        m_items = getUnique(col_value(Cols::item));

        default_random_engine generator(time(nullptr));
        normal_distribution<double> dist_beta(0, std_beta);
        normal_distribution<double> dist_theta(0, std_theta);

        initVectors(dist_theta, m_users, m_theta);
        cout << "Initialized " << m_theta.size() << " users in theta map \n";

        initVectors(dist_beta, m_items, m_beta);
        cout << "Initialized " << m_beta.size() << " items in beta map \n";
    }

    PMF::~PMF()
    {
        stopWorkerThread();
    }

    void PMF::startWorkerThread()
    {
        m_run_compute_loss = true;
        m_loss_thread = thread([this] {
            this->loss();
        });
        cout << "Compute loss thread started. \n";
    }

    void PMF::stopWorkerThread()
    {
        m_run_compute_loss = false;
        if (m_loss_thread.joinable())
        {
            m_loss_thread.join();
            cout << "Compute loss thread stopped. \n";
        }
    }

    // Gets a set of unique int ID values for column col_idx in m_data
    set<int> PMF::getUnique(int col_idx)
    {
        const MatrixXd &col = m_data->col(col_idx);
        set<int> unique{col.data(), col.data() + col.size()};
        return unique;
    }

    // Initializes map vmap for each entity with random vector of size m_k with
    // distribution dist
    void PMF::initVectors(normal_distribution<> &dist, const set<int> &entities,
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
        while (m_run_compute_loss)
        {
            this_thread::sleep_for(chrono::seconds(1));
            lock_guard<mutex> guard(m_mutex);

            double l;
            for (auto &i : m_users)
            {
                l += logNormPDF(m_theta[i]);
            }
            for (auto &j : m_items)
            {
                l += logNormPDF(m_beta[j]);
            }
            for (int idx = 0; idx < m_data->rows(); idx++)
            {
                int i = (*m_data)(idx, 0);
                int j = (*m_data)(idx, 1);
                double r = (*m_data)(idx, 2);
                double r_hat = m_theta[i].dot(m_beta[j]);
                l += logNormPDF(r, r_hat);
            }

            m_losses.push_back(l);
            cout << "[worker] loss: " << l << endl;
        }
    }

    // Subset data by rows where values in column is equal to ID
    MatrixXd PMF::subsetByID(MatrixXd& data, int ID, int column)
    {
        VectorXi idx = (data.col(column).array() == ID).cast<int>();
        MatrixXd submatrix(idx.sum(), data.cols());
        int cur_row = 0;
        for (int i = 0; i < data.rows(); ++i)
        {
            if (idx[i]) {
                submatrix.row(cur_row) = data.row(i);
                cur_row++;
            }
        }
        return submatrix;
    }

    // Approximate gradient of beta from subsample sample
    VectorXd PMF::getBetaGrad(MatrixXd& sample, int j) {
        // extract sub-matrix of item j's data
        const MatrixXd item_data = subsetByID(sample, j, col_value(Cols::item));
        const VectorXi &i_users = item_data.col(col_value(Cols::user)).cast<int>();
        const VectorXd &i_ratings = item_data.col(col_value(Cols::rating));

        // compute gradient update of item attribute vectors
        VectorXd grad = -(1.0 / m_std_beta) * m_beta[j];
        for (int idx = 0; idx < i_users.size(); idx++)
        {
            int i = i_users(idx);
            double rating = i_ratings(idx);
            double rating_hat = m_theta[i].dot(m_beta[j]);
            grad += (rating - rating_hat) * m_theta[i];
        }
    }

    // Approximate gradient of theta from subsample sample
    VectorXd PMF::getThetaGrad(MatrixXd& sample, int i) {
        // extract sub-matrix of user i's sample
        const MatrixXd user_data = subsetByID(sample, i, col_value(Cols::user));
        const VectorXi &j_items = user_data.col(col_value(Cols::item)).cast<int>();
        const VectorXd &j_ratings = user_data.col(col_value(Cols::rating));

        // approximate gradient of user preference vectors
        VectorXd grad = -(1.0 / m_std_theta) * m_theta[i];
        for (int idx = 0; idx < j_items.size(); idx++)
        {
            int j = j_items(idx);
            double rating = j_ratings(idx);
            double rating_hat = m_theta[i].dot(m_beta[j]);
            grad += (rating - rating_hat) * m_beta[j];
        }
        return grad;
    }

    vector<double> PMF::fit(int epochs, double gamma)
    {
        int num_workers = 2;

        startWorkerThread();
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            // TODO: split rows of m_data randomly into separate smaller subsamples
            int n = m_data->rows()/2;
            map<int, MatrixXd> samples;  // subsets of m_data for each worker thread; together covering full dataset
            samples[1] = m_data->topRows(n);  // TODO: create better random sampling scheme
            samples[2] = m_data->bottomRows(n);

            // update theta vectors
            for (int i : m_users)
            {
                lock_guard<mutex> guard(m_mutex);

                // distribute getThetaGrad over worker nodes, one for each subsample of m_data
                VectorXd grad = VectorXd::Zero(m_k);
                for (int w = 0; w < num_workers; w++)  // TODO: for each worker (parallelize this)
                    grad += getThetaGrad(samples[w], i);

                // once all worker nodes have completed, update theta_i
                VectorXd update = m_theta[i] + gamma * grad;
                update.normalize();
                m_theta[i] = update;
            }

            // update beta vectors
            for (int j : m_items)
            {
                lock_guard<mutex> guard(m_mutex);

                // distribute getThetaGrad over worker nodes, one for each subsample of m_data
                VectorXd grad = VectorXd::Zero(m_k);
                for (int w = 0; w < num_workers; w++)  // TODO: for each worker (parallelize this)
                    grad += getBetaGrad(samples[w], j);

                // once all worker nodes have completed, update beta_j
                VectorXd update = m_beta[j] + gamma * grad;
                update.normalize();
                m_beta[j] = update;
            }
        }
        stopWorkerThread();

        // TODO: trigger background thread to compute loss on current m_theta and m_beta using full m_data
        return m_losses;
    }

    // Predict ratings using learned theta and beta vectors in model
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