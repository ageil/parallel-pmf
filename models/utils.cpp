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

    double Utils::rmse(const VectorXd &y, const VectorXd &y_hat)
    {
        Expects(y.size() > 0);
        Expects(y.size() == y_hat.size());

        const VectorXd &err = y - y_hat;
        const VectorXd &sq_err = err.array().square();
        const double &ms_error = sq_err.sum() / y.size();

        return sqrt(ms_error);
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