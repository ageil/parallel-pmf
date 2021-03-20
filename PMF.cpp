#include "PMF.h"

namespace Proj
{

    PMF::PMF(MatrixXd &data, const int k, const double std_beta, const double std_theta)
        : m_data(data), m_k(k), m_std_beta(std_beta), m_std_theta(std_theta)
    {
        cout << "PMF initialized `data` size " << data.rows()
             << " x " << data.cols() << " with k=" << k
             << " std_beta=" << std_beta << " std_theta=" << std_theta
             << endl;
        // default_random_engine generator(time(nullptr));
        // normal_distribution<double> dist_beta(0, std_beta);
        // normal_distribution<double> dist_theta(0, std_theta);
    }

    void PMF::initializeTheta(const vector<int> &users)
    {
        for (const int user : users)
        {
            m_theta[user] = VectorXd(m_k);
        }
    }

    void PMF::initializeBeta(const vector<int> &items)
    {
        for (const int item : items)
        {
            m_beta[item] = VectorXd(m_k);
        }
    }

    double PMF::normPDF(int x, double loc, double scale)
    {
        cerr << "Not implemented yet" << endl;
        return 0;
    }

    double PMF::logNormPDF(int x, double loc, double scale)
    {
        cerr << "Not implemented yet" << endl;
        return 0;
    }

    double PMF::gradLogNormPDF(int x, double loc, double scale)
    {
        cerr << "Not implemented yet" << endl;
        return 0;
    }

    vector<double> PMF::loss(MatrixXd data)
    {
        cerr << "Not implemented yet" << endl;
        return {};
    }

    // `data` is a matrix with m_k rows and 2 columns (user, item).
    // Returns a vector of predictions, getting the dot product of the
    // given user and item from the theta and beta vectors respectively.
    // TODO: need help writing documentation for what this method is doing
    VectorXd PMF::predict(const MatrixXd &data)
    {
        //Should we assert m_k == num_rows here?
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

} //namespace Proj