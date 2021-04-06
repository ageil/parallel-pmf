#ifndef FINAL_PROJECT_DATAMANAGER_H
#define FINAL_PROJECT_DATAMANAGER_H

#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../csvlib/csv.h"

namespace Utils
{
using namespace std;
using namespace Eigen;

struct ItemMap
{
    ItemMap(unordered_map<int, string> in, unordered_map<string, int> ni, unordered_map<int, string> ig,
            unordered_map<string, string> ng, unordered_map<string, unordered_set<int>> gi)
        : id_name(std::move(in)), name_id(std::move(ni)), id_genre(std::move(ig)), name_genre(std::move(ng)),
        genre_ids(std::move(gi)){};

    unordered_map<int, string> id_name;
    unordered_map<string, int> name_id;
    unordered_map<int, string> id_genre;
    unordered_map<string, string> name_genre;
    unordered_map<string, unordered_set<int>> genre_ids;
};

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
    ItemMap loadItemMap(const string &input);
};

} // namespace Utils

#endif // FINAL_PROJECT_DATAMANAGER_H
