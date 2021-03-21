#include "utils.h"
#include "PMF.h"

#include <cmath>
#include <iostream>

#include <gsl/gsl_assert>

namespace Model
{

    double Utils::rmse(const VectorXd &y, const double y_hat)
    {
        const auto denominator = y.size();
        Expects(denominator > 0);

        const double c = 1 / (double)denominator;
        const VectorXd &squared = (y.array() - y_hat).square();
        const double &summed = squared.sum();

        return sqrt(c * summed);
    }

    double Utils::r2(const VectorXd &y, const VectorXd &y_hat)
    {
        //TODO
        cerr << "Not implemented yet" << endl;
        return 0;
    }

    // def R2(y, y_hat):
    // SSE = np.sum((y - y_hat)**2)
    // TSS = np.sum((y - np.mean(y))**2)
    // R2 = 1 - SSE/TSS
    // return R2

    pair<double, double> Utils::topN(const PMF &pmfModel, const MatrixXd &data,
                                     const int N)
    {
        //TODO
        cerr << "Not implemented yet" << endl;
        pair<double, double> p;
        return p;
    }

} // namespace Model