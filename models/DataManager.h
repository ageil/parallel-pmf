#ifndef FINAL_PROJECT_DATAMANAGER_H
#define FINAL_PROJECT_DATAMANAGER_H

#include <Eigen/Dense>
#include <memory>
#include <unordered_map>

#include "../csvlib/csv.h"

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
            unsigned long int getLineNumber(const string &file_name);
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
            unordered_map<int, string> itemIdToName(const string &input);
    };

} //namespace Utils


#endif //FINAL_PROJECT_DATAMANAGER_H
