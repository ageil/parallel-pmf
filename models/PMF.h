#ifndef PMF_H
#define PMF_H

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <utility>

#include <Eigen/Dense>

namespace Model
{
using namespace std;
using namespace Eigen;

struct ThetaBetaSnapshot
{
    ThetaBetaSnapshot(const map<int, VectorXd> theta, const map<int, VectorXd> beta) : theta(theta), beta(beta){};

    const map<int, VectorXd> theta;
    const map<int, VectorXd> beta;
};

struct Metrics
{
    double precision;
    double recall;
};

class PMF
{
  private:
    void initVectors(normal_distribution<> &dist, const vector<int> &entities, map<int, VectorXd> &vmap);
    double logNormPDF(const VectorXd &x, double loc = 0.0, double scale = 1.0);
    double logNormPDF(double x, double loc = 0.0, double scale = 1.0);
    MatrixXd subsetByID(const Ref<MatrixXd> &batch, int ID, int column);
    void compute_loss(const map<int, VectorXd> &theta, const map<int, VectorXd> &beta);
    void compute_loss_from_queue();

    void fitUsers(const Ref<MatrixXd> &batch, const double learning_rate);
    void fitItems(const Ref<MatrixXd> &batch, const double learning_rate);

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

    mutex m_mutex;
    condition_variable m_cv;
    bool m_fit_in_progress;
    queue<ThetaBetaSnapshot> m_loss_queue;

  public:
    PMF(const shared_ptr<MatrixXd> &d, const int k, const double eta_beta, const double eta_theta,
        const vector<int> &users, const vector<int> &items);
    ~PMF();
    vector<double> fit(const int epochs, const double gamma, const int n_threads);
    vector<double> fit_sequential(const int epochs, const double gamma);
    vector<double> fit_parallel(const int epochs, const double gamma, const int n_threads);

    VectorXd predict(const MatrixXd &data);
    VectorXi recommend(int user_id, int N = 10);
    Metrics accuracy(const shared_ptr<MatrixXd> &data, const int N);
};
} // namespace Model

#endif // PMF_H
