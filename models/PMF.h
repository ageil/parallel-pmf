#ifndef PMF_H
#define PMF_H

#include <iostream>
#include <Eigen/Dense>
#include <random>

namespace Model
{
    using namespace std;
    using namespace Eigen;

    class PMF
    {
    private:
        MatrixXd m_data;
        double m_k;
        double m_std_beta;
        double m_std_theta;
        vector<double> m_beta;
        vector<double> m_theta;
        vector<double> m_losses;
        double normPDF(int x, double loc = 0.0, double scale = 1.0);
        double logNormPDF(int x, double loc = 0.0, double scale = 1.0);
        double gradLogNormPDF(int x, double loc = 0.0, double scale = 1.0);
        vector<double> loss(MatrixXd data);

    public:
        PMF(MatrixXd d, int k, double eta_beta, double eta_theta);
        ~PMF();
        double train(int iters);
        MatrixXd predict(MatrixXd data);
        VectorXd recommend(int user);

        //todo: specify return type for "norm_vectors" functions in model.py
    };
} //namespace Model

#endif // PMF_H