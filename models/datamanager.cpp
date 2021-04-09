#include <filesystem>
#include <iostream>
#include <random>
#include <set>

#include "datamanager.h"
#include "utils.h"

#include <gsl/gsl_assert>

namespace DataManager
{

using namespace std;

namespace
{

/**
 * Centers the ratings in the given data matrix such that the midpoint of the rating scale is zero.
 * @param data The matrix to center.
 */
void zeroCenter(MatrixXd &data)
{
    const int rating_col = col_value(Cols::rating);
    set<double> unique_vals{data.col(rating_col).data(), data.col(rating_col).data() + data.col(rating_col).size()};

    double sum = 0;

    for (const auto i : unique_vals)
    {
        sum += i;
    }

    const double mid = sum / unique_vals.size();

    for (int i = 0; i < data.rows(); i++)
    {
        data(i, rating_col) -= mid;
    }
}

/**
 * Randomly shuffles the rows of the given data matrix linearly spaced.
 * @param data The matrix to shuffle the rows of.
 */
void shuffle(MatrixXd &data)
{
    VectorXi ind = VectorXi::LinSpaced(data.rows(), 0, data.rows());
    shuffle(ind.data(), ind.data() + data.rows(), mt19937(random_device()()));
    data = ind.asPermutation() * data;
}

/**
 * Count the total number of lines for the input file.
 * @param file_name Input file name
 * @return Number of lines of the file
 */
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

/**
 * Load the matrix from file and perform preprocessing steps (shuffling rows, zero-centering ratings).
 * @param input_filepath Input file name
 * @return Matrix of the input dataset (dimension: N x 3)
 */
MatrixXd load(const string &input_filepath)
{
    if (!filesystem::exists(input_filepath))
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

    zeroCenter(data);
    shuffle(data);

    return data;
}

} // namespace

/**
 * Initialize DataManager by loading the csv file found in the given input. The data is zero-centered, shuffled, and
 * stored. Additionally, the given ratio will determine how to split the processed data into training data vs. testing
 * data. (e.g. ratio=0.7 will split to 70% train, 30% test)
 * @param input A file path to the csv file of data to load.
 * @param ratio The ratio to split the data into training data vs. testing data.
 */
DataManager::DataManager(const string &input, const double ratio)
    : m_data(make_shared<MatrixXd>(load(input)))

{
    Expects(ratio > 0.0 && ratio <= 1.0);
    tie(m_data_train, m_data_test) = split(ratio);

    // get all user & item ids
    users = Utils::getUnique(m_data, 0);
    items = Utils::getUnique(m_data, 1);
}

/**
 * Splits the m_data rows into a train and test set by ratio (e.g. ratio=0.7 will split to 70% train, 30% test)
 * @param ratio The ratio to split the data into training data vs. testing data.
 * @return A tuple of <MatrixXd, MatrixXd> type, in which the first matrix is the training data and the second matrix is
 * the testing data.
 */
tuple<TrainingData, TestingData> DataManager::split(const double ratio)
{
    Expects(ratio > 0.0 && ratio <= 1.0);
    const int idx = static_cast<int>(m_data->rows() * ratio);

    return {make_shared<MatrixXd>(m_data->topRows(idx)),
            make_shared<MatrixXd>(m_data->bottomRows(m_data->rows() - idx))};
}

/**
 * Gets the training data set.
 * @return A shared_ptr of the matrix of the training data set.
 */
shared_ptr<MatrixXd> DataManager::getTrain() const
{
    return m_data_train;
}

/**
 * Gets the testing data set.
 * @return A shared_ptr of the matrix of the testing data set.
 */
shared_ptr<MatrixXd> DataManager::getTest() const
{
    return m_data_test;
}

/**
 * Load the mappings between items' ID (integer), titles (string), and genres (string)
 * @param input Input file name
 * @return Struct of multiple Maps between ID, titles & genres:
 * ItemMap.id_name - ID->title, ItemMap.name_id - title->ID, ItemMap.id_genre - ID->genre, ItemMap.name_genre -
 * title->genre, Item.genre_ids - genre->Set of IDs of the given genre
 */
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
    unordered_map<int, string> id_name;
    unordered_map<string, int> name_id;
    unordered_map<int, string> id_genre;
    unordered_map<string, string> name_genre;
    unordered_map<string, unordered_set<int>> genre_ids;

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

/**
 * Cast the Cols enum class type to its corresponding enum type
 * @param Cols enum of the Cols type
 * @return Int representing the column idx
 */
int col_value(Cols column)
{
    return static_cast<int>(column);
}

} // namespace DataManager