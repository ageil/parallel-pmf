#ifndef PMF_H
#define PMF_H

#include "datamanager.h"

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
    ThetaBetaSnapshot(const map<int, VectorXd> theta, const map<int, VectorXd> beta)
        : theta(theta)
        , beta(beta){};

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
    // Initializes map vmap for each entity with random vector of size m_k sampling from distribution.
    void initVectors(normal_distribution<> &dist, const vector<int> &entities, map<int, VectorXd> &vmap, const int k);

    // Evaluate log normal PDF at vector x.
    double logNormPDF(const VectorXd &x, double loc = 0.0, double scale = 1.0) const;

    // Evaluate log normal PDF at double x.
    double logNormPDF(double x, double loc = 0.0, double scale = 1.0) const;

    // Subset data by rows where values in column is equal to ID.
    MatrixXd subsetByID(const Ref<MatrixXd> &batch, int ID, int column) const;

    // Calculate the log probability of the data under the current model.
    void compute_loss(const map<int, VectorXd> &theta, const map<int, VectorXd> &beta);

    // Computes loss from the theta and beta snapshots found in the
    // m_loss_queue queue.
    void compute_loss_from_queue();

    // Fit user preference vectors to sample data in batch.
    void fitUsers(const Ref<MatrixXd> &batch, const double learning_rate);

    // Fit item  vectors to sample data in batch.
    void fitItems(const Ref<MatrixXd> &batch, const double learning_rate);

    const shared_ptr<DataManager::DataManager> m_data_mgr;
    const shared_ptr<MatrixXd> m_training_data;
    const double m_std_theta;
    const double m_std_beta;
    map<int, VectorXd> m_theta;
    map<int, VectorXd> m_beta;
    default_random_engine d_generator;

    vector<double> m_losses;
    mutex m_mutex;
    condition_variable m_cv;
    bool m_fit_in_progress;
    queue<ThetaBetaSnapshot> m_loss_queue;

  public:
    PMF(const shared_ptr<DataManager::DataManager> &data_mgr, const int k, const double eta_beta,
        const double eta_theta);
    ~PMF();

    // Fits the ratings data sequentially updating m_theta and m_beta vectors with the learning rate given in gamma.
    // Returns the vector of loss computations computed for every 10 epochs.
    vector<double> fit_sequential(const int epochs, const double gamma);

    // Fits the ratings data in parallel updating m_theta and m_beta vectors with the learning rate given in gamma.
    // This method will divide the ratings data by the given n_thread number of batches.
    // Returns the vector of loss computations computed for every 10 epochs.
    vector<double> fit_parallel(const int epochs, const double gamma, const int n_threads);

    // Predict ratings using learnt theta and beta vectors in model.
    // Input: data matrix with n rows and 2 columns (user, item).
    // Returns a vector of predicted ratings for each user and item.
    VectorXd predict(const MatrixXd &data) const;

    // Recommends top N items for given user_id based on fitted data.
    VectorXi recommend(const int user_id, const int N = 10) const;

    // Return the precision & recall of the top N predicted items for each user in
    // the give dataset.
    Metrics accuracy(const shared_ptr<MatrixXd> &data, const int N) const;
};
} // namespace Model

#endif // PMF_H
