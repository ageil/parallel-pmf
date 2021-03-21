#include "PMF.h"
#include<set>

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

        initializeVectors(dist_beta, m_items, m_beta);
        cout << "Initialized " << m_theta.size() << " users in theta map \n";

        initializeVectors(dist_theta, m_users, m_theta);
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
    void PMF::initializeVectors(normal_distribution<>& dist, set<int>& entities, map<int, VectorXd>& m_vectors)
    {
        auto rand = [&](){ return dist(generator); };
        for (int elem : entities)
        {
            VectorXd vec = VectorXd::NullaryExpr(m_k, rand);
            vec.normalize();
            m_vectors[elem] = vec;
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

    double PMF::fit(int iters, double gamma)
    {
        for (int iter=0; iter<iters; iter++)
        {
            cout << "Iteration: " << iter+1 << "/" << iters << endl;
            for (int i : m_users)
            {
//                double grad = -(1.0 / m_std_theta) * m_theta[i];

            }
            for (int j : m_items)
            {

            }

        }
        double loss = 0.0;
        return loss;
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

} // namespace Model