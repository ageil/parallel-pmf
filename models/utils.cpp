#include "utils.h"
#include "PMF.h"

#include <iostream>
#include <gsl/gsl_assert>

namespace Model
{

    VectorXd Utils::rmse(const double y_hat, const VectorXd &y)
    {
        const double denominator = y.size();
        Expects(denominator > 0);

        const double c = 1 / denominator;

        // sqrt(1/k * sum((y-y_hat^2)))
        VectorXd sub = (c * (y_hat - y.array()).square()).sqrt();
        return sub;
    }

    double Utils::r2(const double y_hat, const VectorXd &y)
    {
        //TODO
        cerr << "Not implemented yet" << endl;
        return 0;
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