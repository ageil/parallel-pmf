#ifndef FINAL_PROJECT_UTILS_H
#define FINAL_PROJECT_UTILS_H

#include <thread>
#include <utility>

#include <boost/regex.hpp>
#include <Eigen/Dense>

namespace Utils
{
using namespace std;
using namespace Eigen;

// Specify argsort option: ascend or descend
enum class Order
{
    ascend = 0,
    descend = 1
};

// Tokenize string delimited by space to vector of double
// Reference:
// https://www.boost.org/doc/libs/1_75_0/doc/html/string_algo/usage.html
vector<string> tokenize(string &s, string delimiter = " ");

// Get all vector indices with non-negative values
vector<int> nonNegativeIdxs(const VectorXd &x);

// Count the total number of intersect elements between two vectors
int countIntersect(const VectorXi &x, const VectorXi &y);

// Get unique int ID values for column col_idx in matrix
vector<int> getUnique(const shared_ptr<MatrixXd> &mat, int col_idx);

// Return the indices that would sort the input vector
// Reference:
// https://stackoverflow.com/questions/25921706/creating-a-vector-of-indices-of-a-sorted-vector
VectorXi argsort(const VectorXd &x, Order option);

// Get root mean squared error between ground-truth (y) and a constant
// prediction (y_hat)
double rmse(const VectorXd &y, double y_hat);

// Get root mean squared error between ground-truth (y) and predictions (y_hat)
double rmse(const VectorXd &y, const VectorXd &y_hat);

// Get coefficient of determination between ground-truth (y) and predictions
// (y_hat)
double r2(const VectorXd &y, const VectorXd &y_hat);

// Calculate cosine similarity between two vectors
double cosine(const VectorXd &v1, const VectorXd &v2);

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

} // namespace Utils

#endif