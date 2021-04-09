#ifndef FINAL_PROJECT_DATAMANAGER_H
#define FINAL_PROJECT_DATAMANAGER_H

#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../csvlib/csv.h"

namespace DataManager
{
using namespace std;
using namespace Eigen;

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

const tuple<string, string, string> Header = {"userId", "itemId", "rating"};

using TrainingData = shared_ptr<MatrixXd>;
using TestingData = shared_ptr<MatrixXd>;

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

class DataManager
{
  private:
    shared_ptr<MatrixXd> m_data;
    shared_ptr<MatrixXd> m_data_train;
    shared_ptr<MatrixXd> m_data_test;

    // Splits m_data into train-test data with given ratio (e.g. ratio=0.7 will split to 70% train, 30% test)
    tuple<TrainingData, TestingData> split(const double ratio);

  public:
    vector<int> users;
    vector<int> items;

    DataManager(const string &input, const double ratio);

    shared_ptr<MatrixXd> getTrain() const;
    shared_ptr<MatrixXd> getTest() const;

    ItemMap loadItemMap(const string &input);
};

} // namespace DataManager

#endif // FINAL_PROJECT_DATAMANAGER_H
