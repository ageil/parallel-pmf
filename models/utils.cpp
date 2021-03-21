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
        double SSE = (y - y_hat).array().square().sum();
        double TSS = (y - y.rowwise().mean()).sum();

        return 1 - (SSE / TSS);
    }

    pair<double, double> Utils::topN(const PMF &pmfModel, const MatrixXd &data,
                                     const int N)
    {
        //TODO
        cerr << "Not implemented yet" << endl;
        pair<double, double> p;
        return p;
    }

} // namespace Model