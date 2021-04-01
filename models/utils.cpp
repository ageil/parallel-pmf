#include "utils.h"
#include "PMF.h"
#include "ratingsdata.h"
#include <cmath>
#include <numeric>
#include <set>
#include <iostream>

#include <gsl/gsl_assert>

namespace Utils
{

    vector<double> positiveIdxs(const VectorXd &x)
    {
        vector<double> indices {};
        for (int i = 0; i < x.size(); i++)
        {
            if (x[i] >= 0)
            {
                indices.push_back(i);
            }
        }

        return indices;
    }

    int countJointIdxs(const VectorXd &x, const VectorXd &y)
    {
        vector<double> vi_x(x.size());
        vector<double> vi_y(y.size());
        VectorXd::Map(vi_x.data(), x.size()) = x;
        VectorXd::Map(vi_y.data(), y.size()) = y;

        vector<double> intersect {};
        set_intersection(vi_x.begin(), vi_x.end(), vi_y.begin(), vi_y.end(), back_inserter(intersect));

        return intersect.size();
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
        double SSE = (y - y_hat).array().square().sum();
        double TSS = (y - y.rowwise().mean()).sum();

        return 1 - (SSE / TSS);
    }

    tuple<double, double> topN(Model::PMF &pmfModel, const shared_ptr<MatrixXd> &data, const int N)
    {
        tuple<double, double> tmp;
        vector<int> likes {};
        vector<int> hits {};

        for (int idx = 0; idx < data->rows(); idx++)
        {
            int user_id = (*data)(idx, 0);
            MatrixXd user_data = pmfModel.subsetByID(*data, user_id, 0);
            VectorXd ratings = user_data.col(2);
            VectorXd items = user_data.col(1);

            // Get all items with positive ratings from the current user id
            vector<double> pos_idxs = positiveIdxs(ratings);
            VectorXd user_liked (pos_idxs.size());
            for (int i = 0; i < pos_idxs.size(); i++) {
                user_liked[i] = items[pos_idxs[i]];
            }

            // Get top N recommendations for the current user_id
            VectorXd rec = pmfModel.recommend(user_id);
            VectorXd top_rec = rec.topRows(N);

            // Get overlap between recommendation & ground-truth "liked"
            int num_liked = pos_idxs.size();
            int num_hits = countJointIdxs(top_rec, user_liked);
            likes.push_back(num_liked);
            hits.push_back(num_hits);
        }

        int n_total_hits = accumulate(hits.begin(), hits.end(), 0);
        int n_total_likes = accumulate(likes.begin(), likes.end(), 0);
        double precision = n_total_hits / (N * data->rows());
        double recall = n_total_hits / n_total_likes;

        return {precision, recall};
    }



} // namespace Utils