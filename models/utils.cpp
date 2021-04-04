#include <cmath>
#include <vector>
#include <set>

#include "utils.h"

#include <gsl/gsl_assert>

namespace Utils
{

    vector<int> nonNegativeIdxs(const VectorXd &x)
    {
        vector<int> indices {};
        for (int i = 0; i < x.size(); i++)
        {
            if (x[i] >= 0)
            {
                indices.push_back(i);
            }
        }

        return indices;
    }

    int countIntersect(const VectorXi &x, const VectorXi &y)
    {
        vector<int> vi_x(x.size());
        vector<int> vi_y(y.size());
        VectorXi::Map(vi_x.data(), x.size()) = x;
        VectorXi::Map(vi_y.data(), y.size()) = y;

        vector<int> intersect {};
        set_intersection(vi_x.begin(), vi_x.end(), vi_y.begin(), vi_y.end(), back_inserter(intersect));

        return intersect.size();
    }

    VectorXi argsort(const VectorXd &x, const Order option)
    {
        Expects(option == Order::ascend or option == Order::descend);

        vector<double> vi (x.size());
        VectorXd::Map(vi.data(), x.size()) = x;

        vector<int> indices (x.size());
        int idx = 0;
        std::generate(indices.begin(), indices.end(), [&] { return idx++; });

        if (option == Order::ascend)
        {
            std::sort(indices.begin(), indices.end(), [&](int a, int b) { return vi[a] < vi[b]; });
        }
        else
        {
            std::sort(indices.begin(), indices.end(), [&](int a, int b) { return vi[a] > vi[b]; });
        }

        Eigen::Map<VectorXi> indices_sorted(indices.data(), indices.size());

        return indices_sorted;
    }

    vector<int> getUnique(const shared_ptr<MatrixXd> &mat, int col_idx)
    {
        const MatrixXd &col = mat->col(col_idx);
        set<int> unique_set{col.data(), col.data() + col.size()};
        vector<int> unique(unique_set.begin(), unique_set.end());

        return unique;
    }

    double rmse(const VectorXd &y, const double y_hat)
    {
        const auto denominator = y.size();
        Expects(denominator > 0);

        const double c = 1 / (double)denominator;
        const VectorXd &squared = (y.array() - y_hat).square();
        const double &summed = squared.sum();

        return sqrt(c * summed);
    }

    double rmse(const VectorXd &y, const VectorXd &y_hat)
    {
        Expects(y.size() > 0);
        Expects(y.size() == y_hat.size());

        const VectorXd &err = y - y_hat;
        const VectorXd &sq_err = err.array().square();
        const double &ms_error = sq_err.sum() / y.size();

        return sqrt(ms_error);
    }

    double r2(const VectorXd &y, const VectorXd &y_hat)
    {
        VectorXd y_mean(y.size());
        y_mean.setConstant(y.mean());
        double SSE = (y - y_hat).array().square().sum();
        double TSS = (y - y_mean).array().square().sum();
        return 1 - (SSE / TSS);
    }

} // namespace Utils