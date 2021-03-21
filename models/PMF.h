#ifndef PMF_H
#define PMF_H

#include <iostream>
#include <map>
#include <random>
#include <set>

#include <Eigen/Dense>

namespace Model
{
        using namespace std;
        using namespace Eigen;

        class PMF
        {
        private:
                set<int> getUnique(int col_idx);
                void initVectors(normal_distribution<>& dist, set<int>& entities, map<int, VectorXd>& m_vectors);
                MatrixXd subsetByID(int ID, int column);
                double logNormPDF(VectorXd x, double loc = 0.0, double scale = 1.0);
                double logNormPDF(double x, double loc = 0.0, double scale = 1.0);
                double loss(MatrixXd data);

                const MatrixXd m_data;
                const int m_k;
                const double m_std_theta;
                const double m_std_beta;
                set<int> m_users;
                set<int> m_items;
                map<int, VectorXd> m_theta;
                map<int, VectorXd> m_beta;
                vector<double> m_losses;
                default_random_engine generator;
        public:
                PMF(const MatrixXd &d, const int k, const double eta_beta, const double eta_theta);
                ~PMF() = default;

                vector<double> fit(int iters, double gamma);
                VectorXd predict(const MatrixXd &data);
                VectorXd recommend(int user);

                //todo: specify return type for "norm_vectors" functions in model.py
        };
} //namespace Model

#endif // PMF_H
