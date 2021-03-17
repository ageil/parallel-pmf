#include "PMF.h"

namespace Proj
{

    PMF::PMF(MatrixXd data, int k, double std_beta, double std_theta)
        : m_data(data), m_k(k), m_std_beta(std_beta), m_std_theta(std_theta)
    {
        default_random_engine generator(time(nullptr));
        normal_distribution<double> dist_beta(0, std_beta);
        normal_distribution<double> dist_theta(0, std_theta);

        for (int i = 0; i < k; i++)
        {
            m_beta.push_back(dist_beta(generator));
            m_theta.push_back(dist_theta(generator));
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

    MatrixXd PMF::predict(MatrixXd data)
    {
        cerr << "Not implemented yet" << endl;
        MatrixXd m(1, 1);
        return m;
    }

    VectorXd PMF::recommend(int user)
    {
        cerr << "Not implemented yet" << endl;
        VectorXd v;
        return v;
    }

} //namespace Proj