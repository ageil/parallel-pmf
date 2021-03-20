#ifndef PMF_H
#define PMF_H

#include <iostream>
#include <map>
#include <random>
#include <Eigen/Dense>
namespace Model
{
        using namespace std;
        using namespace Eigen;

        class PMF
        {
        private:
                void initializeTheta(const VectorXd &users);
                void initializeBeta(const VectorXd &items);
                double normPDF(int x, double loc = 0.0, double scale = 1.0);
                double logNormPDF(int x, double loc = 0.0, double scale = 1.0);
                double gradLogNormPDF(int x, double loc = 0.0, double scale = 1.0);
                vector<double> loss(MatrixXd data);

                const MatrixXd m_data;
                const int m_k;
                const double m_std_theta;
                const double m_std_beta;
                VectorXd m_users;
                VectorXd m_items;
                map<int, VectorXd> m_theta;
                map<int, VectorXd> m_beta;
                map<int, VectorXd> m_losses;

        public:
                PMF(const MatrixXd &d, const int k, const double eta_beta, const double eta_theta);
                ~PMF() = default;

                double train(int iters);
                VectorXd predict(const MatrixXd &data);
                VectorXd recommend(int user);

                //todo: specify return type for "norm_vectors" functions in model.py
        };
} //namespace Model

#endif // PMF_H
