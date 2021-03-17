#ifndef FINAL_PROJECT_PMF_H
#define FINAL_PROJECT_PMF_H

#include <iostream>
#include <Eigen/Dense>
#include <random>

namespace Proj
{
    using namespace std;
    using namespace Eigen;

    class PMF
    {
    private:
        MatrixXd m_data;
        double m_k;
        double m_std_theta;
        double m_std_beta;
        vector<double> m_beta;
        vector<double> m_theta;
        vector<double> m_losses;

    public:
        PMF(MatrixXd d, int k, double eta_beta, double eta_theta);
        ~PMF();
        double normPDF(int x, double loc = 0.0, double scale = 1.0);
        double logNormPDF(int x, double loc = 0.0, double scale = 1.0);
        double gradLogNormPDF(int x, double loc = 0.0, double scale = 1.0);
        vector<double> loss(MatrixXd data);
        MatrixXd predict(MatrixXd data);
        VectorXd recommend(int user);

        //todo: specify return type for "norm_vectors" functions in model.py
    };
} //namespace Proj
