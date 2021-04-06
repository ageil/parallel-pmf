#include <iostream>
#include <random>
#include <set>

#include "datamanager.h"
#include "utils.h"

#include <boost/filesystem.hpp>

namespace DataManager
{

namespace fs = boost::filesystem;
using namespace std;

namespace
{

// Centralize data matrix to mean = median ratings
void centralize(MatrixXd &data)
{
    set<double> unique_vals{data.col(2).data(), data.col(2).data() + data.col(2).size()};

    double sum = 0;

    for (const auto i : unique_vals)
    {
        sum += i;
    }

    const double mid = sum / unique_vals.size();

    for (int i = 0; i < data.rows(); i++)
    {
        data(i, 2) -= mid;
    }
}

// Shuffles data matrix
void shuffle(MatrixXd &data)
{
    VectorXi ind = VectorXi::LinSpaced(data.rows(), 0, data.rows());
    shuffle(ind.data(), ind.data() + data.rows(), mt19937(random_device()()));
    data = ind.asPermutation() * data;
}

// Count total line number of the input file
unsigned long int getLineNumber(const string &file_name)
{
    io::CSVReader<3> in(file_name);
    // Todo: make these customizable?
    in.read_header(io::ignore_extra_column, "userId", "movieId", "rating");
    double col1, col2, col3;
    unsigned long int count = 0;

    while (in.read_row(col1, col2, col3))
    {
        count++;
    }

    return count;
}

// Returns matrix of centralized and shuffled data loaded from given input_filepath csv
MatrixXd load(const string &input_filepath)
{
    if (!fs::exists(input_filepath))
    {
        cerr << "Can't find the given input_filepath file: " << input_filepath << endl;
        exit(1);
    }

    // Get total line number, initialize matrix
    const unsigned long int line_count = getLineNumber(input_filepath);
    MatrixXd data(line_count, 3);

    // Read file content
    cout << "Loading input matrix..." << endl;

    io::CSVReader<3> in(input_filepath);
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

    centralize(data);
    shuffle(data);

    return data;
}

} // namespace

DataManager::DataManager(const string &input, const double ratio)
    : m_data(make_shared<MatrixXd>(load(input)))

{
    tie(m_data_train, m_data_test) = split(ratio);

    // get all user & item ids
    users = Utils::getUnique(m_data, 0);
    items = Utils::getUnique(m_data, 1);
}

tuple<TrainingData, TestingData> DataManager::split(const double ratio)
{
    const int idx = static_cast<int>(m_data->rows() * ratio);

    return {make_shared<MatrixXd>(m_data->topRows(idx)),
            make_shared<MatrixXd>(m_data->bottomRows(m_data->rows() - idx))};
}

shared_ptr<MatrixXd> DataManager::getTrain() const
{
    return m_data_train;
}

shared_ptr<MatrixXd> DataManager::getTest() const
{
    return m_data_test;
}

unordered_map<int, pair<string, string>> DataManager::loadItemNames(const string &input) const
{
    if (!fs::exists(input))
    {
        cerr << "Can't find the given input file: " << input << endl;
        exit(1);
    }

    io::CSVReader<3> in(input);
    in.read_header(io::ignore_extra_column, "movieId", "title", "genres");
    int movie_id;
    string title;
    string genre;
    unordered_map<int, pair<string, string>> id_name;

    while (in.read_row(movie_id, title, genre))
    {
        id_name[movie_id] = pair<string, string>(title, genre);
    }

    return id_name;
}

} // namespace DataManager