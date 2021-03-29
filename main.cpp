#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <tuple>

#include "csvlib/csv.h"
#include "models/PMF.h"
#include "models/utils.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace Model;
using namespace chrono;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Load and centralize rating matrix, split into training & test sets
MatrixXd loadData(const string &input)
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
    MatrixXd ratings(1, 3);

    while (in.read_row(user_id, movie_id, rating))
    {
        Vector3d curr;
        curr << user_id, movie_id, rating;
        ratings.row(ratings.rows() - 1) = curr;
        ratings.conservativeResize(ratings.rows() + 1, ratings.cols());
    }
    ratings.conservativeResize(ratings.rows() - 1, ratings.cols());

    // center ratings to mean = 0
    set<double> unique_ratings{ratings.col(2).data(), ratings.col(2).data() + ratings.col(2).size()};
    double sum = 0;
    for (auto i : unique_ratings)
    {
        sum += i;
    }
    double mid = sum / unique_ratings.size();
    for (int i = 0; i < ratings.rows(); i++)
    {
        ratings(i, 2) -= mid;
    }

    return ratings;
}


// Train-test split
tuple<shared_ptr<MatrixXd>, shared_ptr<MatrixXd>> splitData(shared_ptr<MatrixXd> &mat, double ratio)
{
    VectorXi ind = VectorXi::LinSpaced(mat->rows(), 0, mat->rows());
    shuffle(ind.data(), ind.data() + mat->rows(), std::mt19937(std::random_device()()));
    *mat = ind.asPermutation() * *mat;
    int idx = static_cast<int>(mat->rows() * ratio);
    MatrixXd mat_train = mat->topRows(idx);
    MatrixXd mat_test = mat->bottomRows(mat->rows() - idx);

    return {make_shared<MatrixXd>(mat_train), make_shared<MatrixXd>(mat_test)};
}

// Get unique int ID values for column col_idx in matrix
vector<int> getUnique(shared_ptr<MatrixXd> &mat, int col_idx)
{
    const MatrixXd &col = mat->col(col_idx);
    set<int> unique_set{col.data(), col.data() + col.size()};
    vector<int> unique(unique_set.begin(), unique_set.end());

    return unique;
}

int main(int argc, char **argv)
{
    // parse arguments, path configuration
    string input = "./movielens/ratings.csv";
    fs::path outdir("results");
    int k = 3;
    int n_epochs = 200;  // default # of iterations
    double gamma = 0.01; // default learning rate for gradient descent
    double ratio = 0.7; // train-test split ratio
    int batch_size = 2000;
    int n_threads = 2;

    po::options_description desc("Parameters for Probabilistic Matrix Factorization (PMF)");
    desc.add_options()
        ("help,h", "Help")
        ("input,i", po::value<string>(&input), "Input file name")
        ("output,o", po::value<fs::path>(&outdir), "Output directory\n  [default: current_path/results/]")
        ("n_components,k", po::value<int>(&k), "Number of components (k)\n [default: 3]")
        ("n_epochs,n", po::value<int>(&n_epochs), "Num. of learning iterations\n  [default: 200]")
        ("ratio,r", po::value<double>(&ratio), "Ratio for training/test set splitting\n [default: 0.7]")
        ("gamma", po::value<double>(&gamma), "learning rate for gradient descent\n  [default: 1e-2]")
        ("thread", po::value<int>(&n_threads), "Number of threads for parallelization");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        cout << desc << endl;
        return 0;
    }
    if (outdir.empty())
    {
        outdir = "results";
    }
    if (fs::exists(outdir))
    {
        cout << "Outdir " << outdir << " exists" << endl;
    }
    else
    {
        cout << "Outdir doesn't exist, creating " << outdir << "..." << endl;
        fs::create_directory(outdir);
    }

    // (1). read CSV & split to training & test sets
    shared_ptr<MatrixXd> ratings = make_shared<MatrixXd>(loadData(input));
    vector<int> users = getUnique(ratings, 0);
    vector<int> items = getUnique(ratings, 1);
    auto [ratings_train, ratings_test] = splitData(ratings, ratio);
    const double std_beta = 1.0;
    const double std_theta = 1.0;

    Model::PMF model{ratings_train, k, std_beta, std_theta, users, items};

    // (2). Fit the model to the training data
    auto t0 = chrono::steady_clock::now();
    vector<double> losses = model.fit(n_epochs, gamma, batch_size, n_threads);
    auto t1 = chrono::steady_clock::now();
    double delta_t = std::chrono::duration<double, std::milli> (t1 - t0).count() * 0.001;
    cout << "Running time for " << n_epochs << " iterations: " << delta_t << " s." << endl;
    cout << endl;

    // (3).Evaluate the model on the test data
    VectorXd actual = ratings_test->rightCols(1);
    VectorXd predicted = model.predict(ratings_test->leftCols(2));
    double error = Utils::rmse(actual, predicted);
    double baseline_zero = Utils::rmse(actual, 0.0);
    double baseline_avg = Utils::rmse(actual, actual.mean());
    cout << "RMSE(0): " << baseline_zero << endl;
    cout << "RMSE(mean): " << baseline_avg << endl;
    cout << "RMSE(pred): " << error << endl;

    // (4). TODO: output losses & prediction results to outdir,
    //  write python scripts for visualization & other calculations

    return 0;
}
