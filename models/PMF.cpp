#include "PMF.h"
#include <set>
#include <cmath>
#include <gsl/gsl_assert>

namespace Model
{

    PMF::PMF(const MatrixXd &data, const int k, const double std_beta, const double std_theta)
        : m_data(data), m_k(k), m_std_beta(std_beta), m_std_theta(std_theta) //, m_users(k), m_items(k)
    {
        cout << "Initializing PMF with `data` size " << data.rows()
             << " x " << data.cols() << " with k=" << k
             << " std_beta=" << std_beta << " std_theta=" << std_theta
             << endl;

        m_users = getUnique(0);
        m_items = getUnique(1);

        default_random_engine generator(time(nullptr));
        normal_distribution<double> dist_beta(0, std_beta);
        normal_distribution<double> dist_theta(0, std_theta);

        initVectors(dist_beta, m_items, m_beta);
        cout << "Initialized " << m_theta.size() << " users in theta map \n";

        initVectors(dist_theta, m_users, m_theta);
        cout << "Initialized " << m_beta.size() << " items in beta map \n";
    }

    // Gets a set of unique int ID values for column col_idx in m_data
    set<int> PMF::getUnique(int col_idx)
    {
        MatrixXd col = m_data.col(col_idx);
        set<int> unique{col.data(), col.data() + col.size()};
        return unique;
    }

    // Initializes map m_vectors for each entity with random vector of size m_k with distribution dist
    void PMF::initVectors(normal_distribution<>& dist, set<int>& entities, map<int, VectorXd>& m_vectors)
    {
        auto rand = [&](){ return dist(generator); };
        for (int elem : entities)
        {
            VectorXd vec = VectorXd::NullaryExpr(m_k, rand);
            vec.normalize();
            m_vectors[elem] = vec;
        }
    }

    // Evaluate log normal PDF at vector x
    double PMF::logNormPDF(VectorXd x, double loc, double scale)
    {
        VectorXd vloc = VectorXd::Constant(x.size(), loc);
        double norm = (x - vloc).norm();
        double log_prob = -log(scale);
        log_prob -= 1.0/2.0 * log(2.0 * M_PI);
        log_prob -= 1.0/2.0 * (pow(2, norm) / pow(2, scale));
        return log_prob;
    }

    // Evaluate log normal PDF at double x
    double PMF::logNormPDF(double x, double loc, double scale)
    {
        double diff = x - loc;
        double log_prob = -log(scale);
        log_prob -= 1.0/2.0 * log(2.0 * M_PI);
        log_prob -= 1.0/2.0 * (pow(2.0, diff) / pow(2.0, scale));
        return log_prob;
    }

    // Calculate the log probability of the data under the current model
    double PMF::loss(MatrixXd data)
    {
        double l = 0.0;
        for (auto& i : m_users) {
            l += logNormPDF(m_theta[i]);
        }
        for (auto& j : m_items) {
            l += logNormPDF(m_beta[j]);
        }
        for (int idx=0; idx<data.rows(); idx++) {
            int i = data(idx,0);
            int j = data(idx, 1);
            double r = data(idx, 2);
            double r_hat = m_theta[i].dot(m_beta[j]);
            l += logNormPDF(r, r_hat);
        }
        return l;
    }

    // Subset m_data by rows where values in column is equal to ID
    MatrixXd PMF::subsetByID(int ID, int column)
    {
        VectorXi idx = (m_data.col(column).array() == ID).cast<int>();
        MatrixXd submatrix(idx.sum(), m_data.cols());
        int cur_row = 0;
        for (int i=0; i<m_data.rows(); ++i)
        {
            if (idx[i]) {
                submatrix.row(cur_row) = m_data.row(i);
                cur_row++;
            }
        }
        return submatrix;
    }

    vector<double> PMF::fit(int iters, double gamma)
    {
        for (int iter=1; iter<=iters; iter++)
        {
            // update theta vectors
            for (int i : m_users)
            {
                // extract sub-matrix of user i's data
                MatrixXd user_data = subsetByID(i, 0);
                VectorXi j_items = user_data.col(1).cast<int>();
                VectorXd j_ratings = user_data.col(2);

                // compute gradient update of user preference vectors
                VectorXd grad = -(1.0 / m_std_theta) * m_theta[i];
                for (int idx=0; idx<j_items.size(); idx++)
                {
                    int j = j_items(idx);
                    double rating = j_ratings(idx);
                    double rating_hat = m_theta[i].dot(m_beta[j]);
                    grad += (rating - rating_hat) * m_beta[j];
                }
                VectorXd update = m_theta[i] + gamma * grad;
                update.normalize();
                m_theta[i] = update;
            }

            // update beta vectors
            for (int j : m_items)
            {
                // extract sub-matrix of item j's data
                MatrixXd item_data = subsetByID(j, 1);
                VectorXi i_users = item_data.col(0).cast<int>();
                VectorXd i_ratings = item_data.col(2);

                // compute gradient update of item attribute vectors
                VectorXd grad = -(1.0 / m_std_beta) * m_beta[j];
                for (int idx=0; idx<i_users.size(); idx++)
                {
                    int i = i_users(idx);
                    double rating = i_ratings(idx);
                    double rating_hat = m_theta[i].dot(m_beta[j]);
                    grad += (rating - rating_hat) * m_theta[i];
                }
                VectorXd update = m_beta[j] + gamma * grad;
                update.normalize();
                m_beta[j] = update;
            }
            // compute loss
            if (iter % 10 == 0 || iter == 1) {
                double l = loss(m_data);
                m_losses.push_back(l);
                cout << "Iter " << iter << " loss: " << l << endl;
            }
        }
        return m_losses;
    }

    // `data` is a matrix with m_k rows and 2 columns (user, item).
    // Returns a vector of predictions, getting the dot product of the
    // given user and item from the theta and beta vectors respectively.
    // TODO: need help writing documentation for what this method is doing

    // Predict ratings using learned theta and beta vectors
    // Returns a vector of predicted ratings for each user and item in input data
    VectorXd PMF::predict(const MatrixXd &data)
    {
        Expects(data.cols() == 2);
        const int num_rows = data.rows();

        VectorXd predictions(num_rows);
        for (int i = 0; i < num_rows; ++i)
        {
            int user = data(i, 0);
            int item = data(i, 1);

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