#ifndef FINAL_PROJECT_DATAMANAGER_H
#define FINAL_PROJECT_DATAMANAGER_H

#include <Eigen/Dense>
#include <memory>

namespace Utils
{
using namespace std;
using namespace Eigen;

class DataManager
{
  private:
    shared_ptr<MatrixXd> m_data;
    shared_ptr<MatrixXd> m_data_train;
    shared_ptr<MatrixXd> m_data_test;
    void centralize();
    void split(double ratio);

  public:
    vector<int> users;
    vector<int> items;
    DataManager() = default;
    ~DataManager() = default;
    shared_ptr<MatrixXd> load(const string &input, double ratio);
    shared_ptr<MatrixXd> getTrain();
    shared_ptr<MatrixXd> getTest();
};

} // namespace Utils

#endif // FINAL_PROJECT_DATAMANAGER_H
