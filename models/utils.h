#ifndef FINAL_PROJECT_UTILS_H
#define FINAL_PROJECT_UTILS_H

#include <utility>
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
        VectorXd rmse(const double y, const VectorXd &y_hat);

        // TODO: need docs for r2
        double r2(const double y, const VectorXd &y_hat);

        // TODO: need docs for topN. Returns pair of <presision, recall>
        pair<double, double> topN(const PMF &pmfModel, const MatrixXd &data,
                                  int N = 10);

    } //namepsace Utils
} //namespace Model

#endif