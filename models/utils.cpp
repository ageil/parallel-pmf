#include "utils.h"
#include "../csvlib/csv.h"
#include <filesystem>
#include <iostream>
#include <set>
#include <vector>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <gsl/gsl_assert>

namespace Model
{
namespace Utils
{

/**
 * Tokenize (split) a string into vector of compoments delimited by the same separater (e.g. comma, tab, etc.)
 * @param str Input string to be splitted
 * @param delimiter Separator character
 * @return Vector of tokenized strings
 */
vector<string> tokenize(string &str, const string delimiter)
{
    vector<string> tokenized{};
    boost::split(tokenized, str, boost::is_any_of(delimiter), boost::token_compress_on);

    return tokenized;
}

/**
 * Return vector of non-negative indices from the given vector
 * @param x Vector of doubles
 * @return Indices of non-negative indices of the given vector
 */
vector<int> nonNegativeIdxs(const VectorXd &x)
{
    vector<int> indices{};
    for (int i = 0; i < x.size(); i++)
    {
        if (x[i] >= 0)
        {
            indices.push_back(i);
        }
    }

    return indices;
}

/**
 * Count number of intersect items between two input vectors
 * @param x Vector of integers (Eigen object)
 * @param y Vector of integers (Eigen object)
 * @return Number of common items between vectors x & y
 */
int countIntersect(const VectorXi &x, const VectorXi &y)
{
    vector<int> vi_x(x.size());
    vector<int> vi_y(y.size());
    VectorXi::Map(vi_x.data(), x.size()) = x;
    VectorXi::Map(vi_y.data(), y.size()) = y;

    vector<int> intersect{};
    set_intersection(vi_x.begin(), vi_x.end(), vi_y.begin(), vi_y.end(), back_inserter(intersect));

    return intersect.size();
}

/**
 * Return the indices that would sort the input vector
 * @param x Input vector of doubles (Eigen object)
 * @param option Sorting option: (Order::ascend or Order::descend)
 * @return Vector of indices represent the sorted order of the input vector
 */
VectorXi argsort(const VectorXd &x, const Order option)
{
    Expects(option == Order::ascend or option == Order::descend);

    vector<double> vi(x.size());
    VectorXd::Map(vi.data(), x.size()) = x;

    vector<int> indices(x.size());
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

/**
 * Return the unique items in a given column of the input matrix
 * @param mat Input matrix whose given column's unique values will be extracted
 * @param col_idx Column index of the matrix to calculate unique items
 * @return vector of unique items from the given column of the input matrix
 */
vector<int> getUnique(const MatrixXd &mat, int col_idx)
{
    Expects(col_idx >= 0 and col_idx <= mat.cols());

    const MatrixXd &col = mat.col(col_idx);
    set<int> unique_set{col.data(), col.data() + col.size()};
    vector<int> unique(unique_set.begin(), unique_set.end());

    return unique;
}

/**
 * Calculate root-mean-squared error (RMSE) between a vector and an integer representing the constant prediction
 * @param y Ground-truth vector
 * @param y_hat Integer that represent a vector of constant predictions
 * @return sqrt( ∑_i(y_i - y_hat_i)^2 / size(y))
 */
double rmse(const VectorXd &y, const double y_hat)
{
    const auto denominator = y.size();
    Expects(denominator > 0);

    const double c = 1 / (double)denominator;
    const VectorXd &squared = (y.array() - y_hat).square();
    const double &summed = squared.sum();

    return sqrt(c * summed);
}

/**
 * Calculate root-mean-squared error (RMSE) between two vectors
 * @param y Ground-truth vector
 * @param y_hat Prediction vector
 * @return Rsqrt( ∑_i(y_i - y_hat_i)^2 / size(y))
 */
double rmse(const VectorXd &y, const VectorXd &y_hat)
{
    Expects(y.size() > 0);
    Expects(y.size() == y_hat.size());

    const VectorXd &err = y - y_hat;
    const VectorXd &sq_err = err.array().square();
    const double &ms_error = sq_err.sum() / y.size();

    return sqrt(ms_error);
}

/**
 * Calculate Coefficient of determination between two vectors
 * @param y Ground-truth vector
 * @param y_hat Prediction vector
 * @return  Coefficient of determination between y & y_hat
 */
double r2(const VectorXd &y, const VectorXd &y_hat)
{
    Expects(y.size() > 0);
    Expects(y.size() == y_hat.size());

    VectorXd y_mean(y.size());
    y_mean.setConstant(y.mean());
    double SSE = (y - y_hat).array().square().sum();
    double TSS = (y - y_mean).array().square().sum();
    return 1 - (SSE / TSS);
}

/**
 * Calculate cosine similarity between two vectors
 * @param v1 Vector 1
 * @param v2 Vector 2
 * @return cosine(y1, y2) = (y1 dot y2) / (||y1|| * ||y2||)
 */
double cosine(const VectorXd &v1, const VectorXd &v2)
{
    Expects(v1.size() > 0);
    Expects(v1.size() == v2.size());
    double distance = v1.dot(v2) / (v1.norm() * v2.norm());

    return distance;
}

/**
 * Load the mappings between items' ID (integer), item_names (string), and item_attributes (string)
 * @param input Input file name
 * @return Struct of multiple Maps between ID, item_names & item_attributes:
 * ItemMap.id_name - ID->item_name, ItemMap.name_id - item_name->ID, ItemMap.id_item_attributes -
 * ID->item_attributes, ItemMap.name_item_attributes - item_name->item_attributes, Item.item_attributes_ids -
 * item_attributes->Set of IDs of the given item_attributes
 */
ItemMap loadItemMap(const string &input)
{
    if (!filesystem::exists(input))
    {
        cerr << "Can't find the given input file: " << input << endl;
        exit(1);
    }

    io::CSVReader<3> in(input);

    in.read_header(io::ignore_extra_column, "itemId", "itemName", "itemAttributes");
    int id;
    string item_name;
    string item_attributes;

    unordered_map<int, string> id_name;
    unordered_map<string, int> name_id;
    unordered_map<int, string> id_item_attributes;
    unordered_map<string, string> name_item_attributes;
    unordered_map<string, unordered_set<int>> item_attributes_ids;

    while (in.read_row(id, item_name, item_attributes))
    {
        id_name[id] = item_name;
        name_id[item_name] = id;
        id_item_attributes[id] = item_attributes;
        name_item_attributes[item_name] = item_attributes;
        string first_item_attributes = Utils::tokenize(item_attributes, "|")[0];
        if (item_attributes_ids.find(first_item_attributes) == item_attributes_ids.end())
        {
            unordered_set<int> id_set(id);
            item_attributes_ids[item_attributes] = id_set;
        }
        else
        {
            item_attributes_ids[item_attributes].insert(id);
        }
    }

    return ItemMap{id_name, name_id, id_item_attributes, name_item_attributes, item_attributes_ids};
}

/**
 * Cast the Cols enum class type to its corresponding enum type
 * @param Cols enum of the Cols type
 * @return Int representing the column idx
 */
int col_value(Cols column)
{
    return static_cast<int>(column);
}

} // namespace Utils
} // namespace Model