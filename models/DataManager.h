#ifndef FINAL_PROJECT_DATAMANAGER_H
#define FINAL_PROJECT_DATAMANAGER_H

#include <Eigen/Dense>
#include <memory>
#include <tuple>
#include <unordered_map>

#include "../csvlib/csv.h"

namespace DataManager
{
using namespace std;
using namespace Eigen;

using TrainingData = shared_ptr<MatrixXd>;
using TestingData = shared_ptr<MatrixXd>;

class DataManager
{
  private:
    shared_ptr<MatrixXd> m_data;
    shared_ptr<MatrixXd> m_data_train;
    shared_ptr<MatrixXd> m_data_test;

    tuple<TrainingData, TestingData> split(const double ratio);

  public:
    vector<int> users;
    vector<int> items;

    DataManager(const string &input, const double ratio);

    shared_ptr<MatrixXd> getTrain() const;
    shared_ptr<MatrixXd> getTest() const;
    unordered_map<int, string> itemIdToName(const string &input);
};

} // namespace DataManager

#endif // FINAL_PROJECT_DATAMANAGER_H
