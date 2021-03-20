#ifndef FINAL_PROJECT_PMF_H
#define FINAL_PROJECT_PMF_H

#include <iostream>
#include <map>
#include <random>

#include <Eigen/Dense>

namespace Proj
{
    using namespace std;
    using namespace Eigen;

    class PMF
    {
    private:
        void initializeTheta(const vector<int> &users);
        void initializeBeta(const vector<int> &items);

        MatrixXd m_data;
        const int m_k;
        double m_std_theta;
        double m_std_beta;
        map<int, VectorXd> m_beta;
        map<int, VectorXd> m_theta;
        map<int, VectorXd> m_losses;

    public:
        PMF(MatrixXd &d, const int k, double eta_beta, double eta_theta);
        ~PMF() = default;

        double normPDF(int x, double loc = 0.0, double scale = 1.0);
        double logNormPDF(int x, double loc = 0.0, double scale = 1.0);
        double gradLogNormPDF(int x, double loc = 0.0, double scale = 1.0);
        vector<double> loss(MatrixXd data);
        VectorXd predict(const MatrixXd &data);
        VectorXd recommend(int user);

        //todo: specify return type for "norm_vectors" functions in model.py
    };
} //namespace Proj

#endif