#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>

#include "models/DataManager.h"
#include "models/PMF.h"
#include "models/utils.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace Model;
using namespace Utils;
using namespace chrono;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char **argv) {
  // Initialize default values for arguments, path configuration
  string input = "./movielens/ratings.csv";
  fs::path outdir("results");
  int k = 3;
  int n_epochs = 200;  // default # of iterations
  double gamma = 0.01; // default learning rate for gradient descent
  double ratio = 0.7;  // train-test split ratio
  int batch_size = 2000;

  double std_theta = 1.0;
  double std_beta = 1.0;

  po::options_description desc(
      "Parameters for Probabilistic Matrix Factorization (PMF)");
  desc.add_options()("help,h", "Help")("input,i", po::value<string>(&input),
                                       "Input file name")(
      "output,o", po::value<fs::path>(&outdir),
      "Output directory\n  [default: current_path/results/]")(
      "n_components,k", po::value<int>(&k),
      "Number of components (k)\n [default: 3]")(
      "n_epochs,n", po::value<int>(&n_epochs),
      "Num. of learning iterations\n  [default: 200]")(
      "ratio,r", po::value<double>(&ratio),
      "Ratio for training/test set splitting\n [default: 0.7]")(
      "batch_size,b", po::value<int>(&batch_size),
      "Number of batch size for parallelization")(
      "gamma", po::value<double>(&gamma),
      "learning rate for gradient descent\n  [default: 2000]")(
      "std_theta", po::value<double>(&std_theta),
      "Std. of theta's prior normal distribution\n  [default: 1]")(
      "std_beta", po::value<double>(&std_beta),
      "Std. of beta's prior normal distribution\n  [default: 1]");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }
  if (outdir.empty()) {
    outdir = "results";
  }
  if (fs::exists(outdir)) {
    cout << "Outdir " << outdir << " exists" << endl;
  } else {
    cout << "Outdir doesn't exist, creating " << outdir << "..." << endl;
    fs::create_directory(outdir);
  }

  // (1). read CSV & split to training & test sets
  Utils::DataManager dm = DataManager();
  shared_ptr<MatrixXd> ratings = dm.load(input, ratio);
  shared_ptr<MatrixXd> ratings_train = dm.getTrain();
  shared_ptr<MatrixXd> ratings_test = dm.getTest();

  // (2). Fit the model to the training data
  auto t0 = chrono::steady_clock::now();
  Model::PMF model{dm.getTrain(), k, std_beta, std_theta, dm.users, dm.items};
  vector<double> losses = model.fit(n_epochs, gamma, batch_size);
  auto t1 = chrono::steady_clock::now();
  double delta_t =
      std::chrono::duration<double, std::milli>(t1 - t0).count() * 0.001;
  cout << "Running time for " << n_epochs << " iterations: " << delta_t << " s."
       << endl;
  cout << endl;

  // (3).Evaluate the model on the test data
  // RMSE of baseline, avg. & the learned model
  VectorXd actual = ratings_test->rightCols(1);
  VectorXd predicted = model.predict(ratings_test->leftCols(2));
  double error = Utils::rmse(actual, predicted);
  double baseline_zero = Utils::rmse(actual, 0.0);
  double baseline_avg = Utils::rmse(actual, actual.mean());

  // precision & recall of the top N items recommended for each user
  int N = 500;
  Metrics acc = model.accuracy(ratings_train, N);

  cout << "RMSE(0): " << baseline_zero << endl;
  cout << "RMSE(mean): " << baseline_avg << endl;
  cout << "RMSE(pred): " << error << endl;
  cout << "Metrics(pred) for top " << N << " recommended items for each user\n"
       << "Precision: " << acc.precision << " Recall: " << acc.recall << endl;

  // (4). Output losses & prediction results to outdir,

  return 0;
}
