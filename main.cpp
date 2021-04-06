#include <chrono>
#include <iostream>
#include <memory>

#include "models/PMF.h"
#include "models/datamanager.h"
#include "models/utils.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace Model;
using namespace Utils;
using namespace chrono;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char **argv)
{
    // Initialize default values for arguments, path configuration
    string input = "./movielens/ratings.csv";
    string item_name_input = "./movielens/movies.csv";
    fs::path outdir("results");
    int k = 3;
    int n_epochs = 200;  // default # of iterations
    double gamma = 0.01; // default learning rate for gradient descent
    double ratio = 0.7;  // train-test split ratio
    int n_threads = 50;

    double std_theta = 1.0;
    double std_beta = 1.0;

    po::options_description desc("Parameters for Probabilistic Matrix Factorization (PMF)");
    desc.add_options()("help,h", "Help")("input,i", po::value<string>(&input), "Input file name")(
        "output,o", po::value<fs::path>(&outdir), "Output directory\n  [default: current_path/results/]")(
        "n_components,k", po::value<int>(&k), "Number of components (k)\n [default: 3]")(
        "n_epochs,n", po::value<int>(&n_epochs), "Num. of learning iterations\n  [default: 200]")(
        "ratio,r", po::value<double>(&ratio), "Ratio for training/test set splitting\n [default: 0.7]")(
        "thread", po::value<int>(&n_threads), "Number of threads for parallelization")(
        "gamma", po::value<double>(&gamma), "learning rate for gradient descent\n  [default: 2000]")(
        "std_theta", po::value<double>(&std_theta), "Std. of theta's prior normal distribution\n  [default: 1]")(
        "std_beta", po::value<double>(&std_beta), "Std. of beta's prior normal distribution\n  [default: 1]")(
        "run_sequential,s", po::bool_switch()->default_value(false), "Enable running model fitting sequentially");

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

    const bool run_fit_sequential = vm["run_sequential"].as<bool>();

    // (1). read CSV & split to training & test sets
    auto dm_t0 = chrono::steady_clock::now();

    const auto dataManager = make_shared<DataManager::DataManager>(input, ratio);

    auto dm_t1 = chrono::steady_clock::now();
    double dm_delta_t = std::chrono::duration<double, std::milli>(dm_t1 - dm_t0).count() * 0.001;
    cout << "Took " << dm_delta_t << " s. to load data. \n\n";

    // (2). Fit the model to the training data
    Model::PMF model{dataManager, k, std_beta, std_theta};

    auto fit_t0 = chrono::steady_clock::now();
    vector<double> losses;

    if (run_fit_sequential)
    {
        losses = model.fitSequential(n_epochs, gamma);
    }
    else
    {
        losses = model.fitParallel(n_epochs, gamma, n_threads);
    }

    auto fit_t1 = chrono::steady_clock::now();
    double fit_delta_t = std::chrono::duration<double, std::milli>(fit_t1 - fit_t0).count() * 0.001;
    cout << "Running time for " << n_epochs << " iterations: " << fit_delta_t << " s.\n\n";

    // (3).Evaluate the model on the test data
    // RMSE of baseline, avg. & the learned model
    shared_ptr<MatrixXd> ratings_test = dataManager->getTest();
    VectorXd actual = ratings_test->rightCols(1);
    VectorXd predicted = model.predict(ratings_test->leftCols(2));
    double error = Utils::rmse(actual, predicted);
    double baseline_zero = Utils::rmse(actual, 0.0);
    double baseline_avg = Utils::rmse(actual, actual.mean());

    cout << "RMSE(0): " << baseline_zero << endl;
    cout << "RMSE(mean): " << baseline_avg << endl;
    cout << "RMSE(pred): " << error << endl;

    // precision & recall of the top N items recommended for each user [Not in use]
    /*
    int N = 500;
    Metrics acc = model.accuracy(ratings_train, N);
    cout << "Metrics(pred) for top " << N << " recommended items for each user\n"
         << "Precision: " << acc.precision << " Recall: " << acc.recall << endl;
     */

    // (4). Recommend top 10 movies for a user
    int user_id = 120;
    int n_top_items = 10;

    // item id <-> name map
    unordered_map<int, pair<string, string>> item_map = dataManager->loadItemNames(item_name_input);
    vector<pair<string, string>> rec = model.recommend(user_id, item_map, n_top_items);

    cout << "\nTop 10 recommended movies for user " << user_id << " :" << endl;
    for (auto &info : rec)
    {
        cout << "Movie: " << info.first << '\t' << "Genre: " << info.second << endl;
    }

    // (5). Output losses & prediction results to outdir

    return 0;
}
