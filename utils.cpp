#include "utils.h"
#include "PMF.h"

#include <iostream>

namespace Proj
{

    VectorXd Utils::rmse(const double y, const VectorXd &y_hat)
    {
        //TODO
        cerr << "Not implemented yet" << endl;
        const int k = y_hat.size();

        return VectorXd(k);
    }

    double Utils::r2(const double y, const VectorXd &y_hat)
    {
        //TODO
        cerr << "Not implemented yet" << endl;
        return 0;
    }

    pair<double, double> Utils::topN(const PMF &pmfModel, const MatrixXd &data,
                                     int N)
    {
        //TODO
        cerr << "Not implemented yet" << endl;
        pair<double, double> p;
        return p;
    }

} // namespace Proj