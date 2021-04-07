#ifndef FINAL_PROJECT_UTILS_H
#define FINAL_PROJECT_UTILS_H

#include <thread>
#include <utility>

#include <Eigen/Dense>
#include <boost/regex.hpp>

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

// Reference:
// https://www.boost.org/doc/libs/1_75_0/doc/html/string_algo/usage.html
vector<string> tokenize(string &s, string delimiter = " ");

vector<int> nonNegativeIdxs(const VectorXd &x);

int countIntersect(const VectorXi &x, const VectorXi &y);

vector<int> getUnique(const shared_ptr<MatrixXd> &mat, int col_idx);

// Reference:
// https://stackoverflow.com/questions/25921706/creating-a-vector-of-indices-of-a-sorted-vector
VectorXi argsort(const VectorXd &x, Order option);

double rmse(const VectorXd &y, double y_hat);

double rmse(const VectorXd &y, const VectorXd &y_hat);

double r2(const VectorXd &y, const VectorXd &y_hat);

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