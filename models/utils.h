#ifndef FINAL_PROJECT_UTILS_H
#define FINAL_PROJECT_UTILS_H

#include <Eigen/Dense>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace Model
{
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

vector<int> getUnique(const MatrixXd &mat, int col_idx);

// Reference:
// https://stackoverflow.com/questions/25921706/creating-a-vector-of-indices-of-a-sorted-vector
VectorXi argsort(const VectorXd &x, Order option);

double rmse(const VectorXd &y, double y_hat);

double rmse(const VectorXd &y, const VectorXd &y_hat);

double r2(const VectorXd &y, const VectorXd &y_hat);

double cosine(const VectorXd &v1, const VectorXd &v2);

// Reference:
// guarded_thread from Prof. Stroustrup's "Concurrency and Parallelism" lecture - slide 37.
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

struct ItemMap
{
    ItemMap(unordered_map<int, string> in, unordered_map<string, int> ni, unordered_map<int, string> ig,
            unordered_map<string, string> ng, unordered_map<string, unordered_set<int>> gi)
        : id_name(std::move(in))
        , name_id(std::move(ni))
        , id_item_attributes(std::move(ig))
        , name_item_attributes(std::move(ng))
        , item_attributes_ids(std::move(gi)){};

    unordered_map<int, string> id_name;
    unordered_map<string, int> name_id;
    unordered_map<int, string> id_item_attributes;
    unordered_map<string, string> name_item_attributes;
    unordered_map<string, unordered_set<int>> item_attributes_ids;
};

ItemMap loadItemMap(const string &input);

/**
 * Enums of the columns to their column indices of ratings dataset
 */
enum class Cols
{
    user = 0,
    item = 1,
    rating = 2
};

int col_value(Cols);

} // namespace Utils
} // namespace Model

#endif