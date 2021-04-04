#include <iostream>
#include <set>
#include <random>

#include "DataManager.h"
#include "utils.h"
#include "../csvlib/csv.h"

#include <boost/filesystem.hpp>

namespace Utils
{
    namespace fs = boost::filesystem;
    using namespace std;

    // Load input data from file into matrix
    shared_ptr<MatrixXd> DataManager::load(const string &input, double ratio)
    {
        if (!fs::exists(input))
        {
            cerr << "Can't find the given input file: " << input << endl;
            exit(1);
        }
        cout << "Loading input matrix..." << endl;
        io::CSVReader<3> in(input);
        in.read_header(io::ignore_extra_column, "userId", "movieId", "rating");
        int user_id;
        int movie_id;
        double rating;
        MatrixXd data(1, 3);

        while (in.read_row(user_id, movie_id, rating))
        {
            Vector3d curr;
            curr << user_id, movie_id, rating;
            data.row(data.rows() - 1) = curr;
            data.conservativeResize(data.rows() + 1, data.cols());
        }
        data.conservativeResize(data.rows() - 1, data.cols());
        m_data = make_shared<MatrixXd>(data);

        // Centralize & split the data
        centralize();
        split(ratio);

        // get all user & item ids
        users = Utils::getUnique(m_data, 0);
        items = Utils::getUnique(m_data, 1);

        return m_data;
    }


    // centralize data matrix to mean = median ratings
    void DataManager::centralize()
    {
        set<double> unique_vals{m_data->col(2).data(), m_data->col(2).data() + m_data->col(2).size()};
        double sum = 0;
        for (auto i : unique_vals)
        {
            sum += i;
        }
        double mid = sum / unique_vals.size();
        cout << "mid: " << mid << endl;
        for (int i = 0; i < m_data->rows(); i++)
        {
            (*m_data)(i, 2) -= mid;
            if (i < 30) {
                cout << (*m_data)(i, 2) << endl;
            }
        }
    }

    // Train-test split with give ratio (e.g. 70% train, 30% test)
    void DataManager::split(double ratio)
    {
        VectorXi ind = VectorXi::LinSpaced(m_data->rows(), 0,m_data->rows());
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

} //namespace Utils