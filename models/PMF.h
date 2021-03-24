#ifndef PMF_H
#define PMF_H

#include <atomic>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <thread>

#include <Eigen/Dense>

namespace Model
{
        using namespace std;
        using namespace Eigen;

        class PMF
        {
        private:
                set<int> getUnique(int col_idx);
                void initVectors(normal_distribution<> &dist, const vector<int> &entities, map<int, VectorXd> &vmap);
                MatrixXd subsetByID(int ID, int column);
                double logNormPDF(const VectorXd &x, double loc = 0.0, double scale = 1.0);
                double logNormPDF(double x, double loc = 0.0, double scale = 1.0);
                void loss();
                void startWorkerThread();
                void stopWorkerThread();

                void fitUsers(const double gamma, const int start, const int end);
                void fitItems(const double gamma, const int start, const int end);

                const shared_ptr<MatrixXd> m_data;
                const int m_k;
                const double m_std_theta;
                const double m_std_beta;
                vector<int> m_users;
                vector<int> m_items;
                map<int, VectorXd> m_theta;
                map<int, VectorXd> m_beta;
                vector<double> m_losses;
                default_random_engine d_generator;

                atomic_bool m_run_compute_loss;
                thread m_loss_thread;

                mutex m_mutex;

        public:
                PMF(const shared_ptr<MatrixXd> &d, const int k, const double eta_beta, const double eta_theta);
                ~PMF();

                vector<double> fit(const int epochs, const double gamma, const int num_threads);
                VectorXd predict(const MatrixXd &data);
                VectorXd recommend(int user);

                //todo: specify return type for "norm_vectors" functions in model.py
        };
} //namespace Model

#endif // PMF_H
