#include <filesystem>
#include <iostream>
#include <random>
#include <set>

#include "DataManager.h"
#include "utils.h"

namespace Utils
{
using namespace std;

// Load input data from file into matrix
shared_ptr<MatrixXd> DataManager::load(const string &input, double ratio)
{
    if (!filesystem::exists(input))
    {
        cerr << "Can't find the given input file: " << input << endl;
        exit(1);
    }

    // Get total line number, initialize matrix
    unsigned long int line_count = getLineNumber(input);
    MatrixXd data(line_count, 3);

    // Read file content
    cout << "Loading input matrix..." << endl;
    io::CSVReader<3> in(input);
    in.read_header(io::ignore_extra_column, "userId", "movieId", "rating");
    int user_id;
    int movie_id;
    double rating;

    int row_idx = 0;
    while (in.read_row(user_id, movie_id, rating))
    {
        Vector3d curr;
        curr << user_id, movie_id, rating;
        data.row(row_idx) = curr;
        row_idx++;
    }
    m_data = make_shared<MatrixXd>(data);

    // Centralize & split the data
    centralize();
    split(ratio);

    // get all user & item ids
    users = Utils::getUnique(m_data, 0);
    items = Utils::getUnique(m_data, 1);

    return m_data;
}

// Count total line number of the input file
unsigned long int DataManager::getLineNumber(const string &file_name)
{
    io::CSVReader<3> in(file_name);
    in.read_header(io::ignore_extra_column, "userId", "movieId", "rating");
    double col1, col2, col3;
    unsigned long int count = 0;

    while (in.read_row(col1, col2, col3))
    {
        count++;
    }

    return count;
}

// Centralize data matrix to mean = median ratings
void DataManager::centralize()
{
    set<double> unique_vals{m_data->col(2).data(), m_data->col(2).data() + m_data->col(2).size()};
    double sum = 0;
    for (auto i : unique_vals)
    {
        sum += i;
    }
    double mid = sum / unique_vals.size();
    for (int i = 0; i < m_data->rows(); i++)
    {
        (*m_data)(i, 2) -= mid;
    }
}

// Train-test split with give ratio (e.g. 70% train, 30% test)
void DataManager::split(double ratio)
{
    VectorXi ind = VectorXi::LinSpaced(m_data->rows(), 0, m_data->rows());
    std::shuffle(ind.data(), ind.data() + m_data->rows(), std::mt19937(std::random_device()()));
    *m_data = ind.asPermutation() * (*m_data);

    int idx = static_cast<int>(m_data->rows() * ratio);
    m_data_train = make_shared<MatrixXd>(m_data->topRows(idx));
    m_data_test = make_shared<MatrixXd>(m_data->bottomRows(m_data->rows() - idx));
}

shared_ptr<MatrixXd> DataManager::getTrain()
{
    return m_data_train;
}

shared_ptr<MatrixXd> DataManager::getTest()
{
    return m_data_test;
}

ItemMap DataManager::loadItemMap(const string &input)
{
    if (!filesystem::exists(input))
    {
        cerr << "Can't find the given input file: " << input << endl;
        exit(1);
    }

    io::CSVReader<3> in(input);
    in.read_header(io::ignore_extra_column, "movieId", "title", "genres");
    int id;
    string title;
    string genre;
    unordered_map<int, string> id_name{};
    unordered_map<string, int> name_id{};
    unordered_map<int, string> id_genre{};
    unordered_map<string, string> name_genre{};
    unordered_map<string, unordered_set<int>> genre_ids{};

    while (in.read_row(id, title, genre))
    {
        id_name[id] = title;
        name_id[title] = id;
        id_genre[id] = genre;
        name_genre[title] = genre;
        string first_genre = Utils::tokenize(genre, "|")[0];
        if (genre_ids.find(first_genre) == genre_ids.end())
        {
            unordered_set<int> id_set(id);
            genre_ids[genre] = id_set;
        }
        else
        {
            genre_ids[genre].insert(id);
        }
    }

    ItemMap item_map = ItemMap(id_name, name_id, id_genre, name_genre, genre_ids);

    return item_map;
}

} // namespace Utils