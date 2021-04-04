#include <iostream>
#include <cmath>
#include "../models/utils.h"

#include<gsl/gsl_assert>

using namespace std;
using namespace Eigen;

void test_nonNegativeIdxs()
{
    cout << "Testing nonNegativeIdxs..." << endl;
    VectorXd x(7);
    x << -1,-2,-3,0,1,2,3;

    vector<int> pos_idxs_gt {3,4,5,6}; // ground truth
    vector<int> pos_idxs = Utils::nonNegativeIdxs(x);
    Expects(std::equal(pos_idxs_gt.begin(), pos_idxs_gt.end(), pos_idxs.begin()) &&
            pos_idxs_gt.size() == pos_idxs.size());
}

void test_countIntersect()
{
    cout << "Testing countIntersect..." << endl;
    VectorXi x1(6);
    VectorXi x2(6);
    x1 << 0,1,2,3,4,5;
    x2 << 3,4,6,7,8,9;

    int num_intersect_gt = 2; // intersect elements: {3,4}
    int num_intersect = Utils::countIntersect(x1, x2);
    Expects(num_intersect_gt == num_intersect);
}

void test_argsort()
{
    cout << "Testing argsort..." << endl;
    VectorXd x(5);
    x << 1,5,3,2,4;

    VectorXi idxs_ascend_gt(5);
    idxs_ascend_gt << 0,3,2,4,1;
    VectorXi idxs_descend_gt = idxs_ascend_gt.reverse();

    // test ascending argsort
    cout << " - Testing ascending order" << endl;
    VectorXi idxs_ascend = Utils::argsort(x, Utils::Order::ascend);
    Expects(idxs_ascend_gt.isApprox(idxs_ascend));

    // test descending argsort
    cout << " - Testing descending order" << endl;
    VectorXi idxs_descend = Utils::argsort(x, Utils::Order::descend);
    Expects(idxs_descend_gt.isApprox(idxs_descend));
}

void test_rmse()
{
    cout << "Testing rmse..." << endl;
    VectorXd y(5);
    y << 1,2,3,4,5; // ground-truth variables
    double eps = 1e-5;

    // Testing root-mean-squared-error with const. prediction
    // rmse(const VectorXd &y, double y_hat)
    cout << " - Testing rmse with constant prediction (y_hat)..." << endl;
    double y_hat_1 = 3;
    double root_mean_squared_err_gt = sqrt(2);  // sqrt[((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5]
    double root_mean_squared_err_pred = Utils::rmse(y, y_hat_1);
    Expects(abs(root_mean_squared_err_gt - root_mean_squared_err_pred) <= eps);

    // Testing root-mean-squared-error with vector prediction
    // rmse(const VectorXd &y, const VectorXd &y_hat)
    cout << " - Testing rmse with vector prediction (y_hat)..." << endl;
    VectorXd y_hat_2(5);
    y_hat_2 << 1,2,4,4,5; // prediction
    root_mean_squared_err_gt = sqrt(0.2); // sqrt[((1-1)^2 + (2-2)^2 + (3-4)^2 + (4-4)^2 + (5-5)^2) / 5]
    root_mean_squared_err_pred = Utils::rmse(y, y_hat_2);
    Expects(abs(root_mean_squared_err_gt - root_mean_squared_err_pred) <= eps);
}

void test_r2()
{
    cout << "Testing r2..." << endl;
    VectorXd y(5);
    VectorXd y_hat_baseline(5);
    VectorXd y_hat_perfect(5);
    y << 1,2,3,4,5; // ground-truth variables
    y_hat_baseline << 3,3,3,3,3; // baseline prediction - r^2 = 0
    y_hat_perfect << 1,2,3,4,5; // perfect predicetion - r^2 = 1
    double eps= 1e-5;

    double r2_baseline = Utils::r2(y, y_hat_baseline);
    double r2_perfect = Utils::r2(y, y_hat_perfect);
    Expects(abs(r2_baseline - 0.0) <= eps && abs(r2_perfect - 1.0) <= eps);
}

int main()
{
    cout << "Unit test for utility functions:" << endl;
    cout << "-----------------------------------------" << endl;
    test_nonNegativeIdxs();
    test_countIntersect();
    test_argsort();
    test_rmse();
    test_r2();
    cout << "Unit test for utility functions completed" << endl;
    cout << "-----------------------------------------" << endl;

    return 0;
}

