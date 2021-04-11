#ifndef FINAL_PROJECT_UTILS_H
#define FINAL_PROJECT_UTILS_H

#include <filesystem>
#include <thread>
#include <utility>

#include <Eigen/Dense>
#include <boost/program_options.hpp>


namespace Utils
{
using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

/*---   Argument-parsing utility functions    ---*/

// Enums of the possible user options for recommendations
    enum class RecOption
    {
        user = 0,
        item = 1
    };

// Struct to store program arguments
struct Arguments
{
    // Input/output directories
    string indir;
    string mapdir;
    filesystem::path outdir;

    // Model task options
    string task;
    RecOption rec_option;

    // Model parallelization options
    bool run_fit_sequential;
    int n_threads;

    // Model specification parameters
    int k;
    int n_epochs;
    double gamma;
    double ratio;
    double std_theta;
    double std_beta;
    int loss_interval;
};

// Read and parse command-line arguments
Arguments ArgParser(int argc, char** argv, po::variables_map &vm, po::options_description &desc);

// Configure input directories
void configureInput(po::variables_map &vm, Arguments &args);

// Configure output directory
void configureOutput(po::variables_map &vm, Arguments &args);

// Configure task option
void configureTask(po::variables_map &vm, Arguments &args);

// Configure model options
void configurePMF(po::variables_map &vm, Arguments &args);

/*---   Model-fitting utility functions     ---*/

// Specify argsort option: ascend or descend
enum class Order
{
    ascend = 0,
    descend = 1
};

// Reference:
// https://www.boost.org/doc/libs/1_75_0/doc/html/string_algo/usage.html
vector<string> tokenize(string &s, string delimiter = " ");

vector<int> nonNegativeIdxs(const VectorXd &x);

int countIntersect(const VectorXi &x, const VectorXi &y);

vector<int> getUnique(const shared_ptr<MatrixXd> &mat, int col_idx);

// Reference:
// https://stackoverflow.com/questions/25921706/creating-a-vector-of-indices-of-a-sorted-vector
VectorXi argsort(const VectorXd &x, Order option);

double rmse(const VectorXd &y, double y_hat);

double rmse(const VectorXd &y, const VectorXd &y_hat);

double r2(const VectorXd &y, const VectorXd &y_hat);

double cosine(const VectorXd &v1, const VectorXd &v2);

// Reference:
// guarded_thread from Prof. Stroustrup's "Concurrency and Parallelism" lecture - slide 37.
struct guarded_thread : std::thread
{
    using std::thread::thread;

    guarded_thread(const guarded_thread &) = delete;

    guarded_thread(guarded_thread &&) = default;

    ~guarded_thread()
    {
        if (joinable())
            join();
    }
};

} // namespace Utils

#endif