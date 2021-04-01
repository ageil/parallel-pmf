#ifndef FINAL_PROJECT_UTILS_H
#define FINAL_PROJECT_UTILS_H

#include <utility>
#include <thread>
#include <Eigen/Dense>

namespace Model
{
    class PMF; // Forward declare
}

namespace Model
{
    namespace Utils
    {
        using namespace std;
        using namespace Eigen;

        // Get root mean squared error. Returns VectorXd of the same shape as
        // y_hat.
        double rmse(const VectorXd &y, const double y_hat);
        double rmse(const VectorXd &y, const VectorXd &y_hat);

        // TODO: need docs for r2
        double r2(const VectorXd &y, const VectorXd &y_hat);

        // TODO: need docs for topN. Returns pair of <precision, recall>
        pair<double, double> topN(const PMF &pmfModel, const MatrixXd &data, const int N = 10);

        struct guarded_thread : std::thread
        {
            using std::thread::thread;

            guarded_thread(const guarded_thread &) = delete;

            guarded_thread(guarded_thread &&) = default;

            ~guarded_thread()
            {
                if (joinable())
                    join();
            }
        };

    } //namepsace Utils
} //namespace Model

#endif