#include <cmath>
#include <iostream>
#include <set>
#include <vector>

#include "utils.h"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <gsl/gsl_assert>

namespace Utils
{

/**
 * Parse program arguments from terminal and perform pre-condition check
 * @return args Parallel-PMF program arguments
 */
Arguments ArgParser(int argc, char** argv, po::variables_map &vm, po::options_description &desc)
{
    Arguments args;

    configureInput(vm, args);
    configureOutput(vm, args);
    configureTask(vm, args);
    configurePMF(vm, args);

    return args;
}

/**
 * Configure & assign input directories
 * @param vm Command-line variable map
 * @param args Parallel-PMF program arguments
 */
void configureInput(po::variables_map &vm, Arguments &args)
{
    // Configure input directory
    if (vm["use_defaults"].as<bool>())
    {
        args.indir = "./movielens/ratings.csv";
        args.mapdir = "./movielens/movies.csv";
        cout << "[ use_defaults, d ] enabled. Setting indir as " << args.indir
             << " and mapdir as " << args.mapdir << endl;
    }
    else
    {
        if (!vm.count("input"))
        {
            cerr << "[ input, i ] not provided. Please provide an input file or specify -d to use defaults\n";
            exit(1);
        }

        if (!vm.count("map"))
        {
            cerr << "[ map, m ] not provided. Please provide an map input file or specify -d to use defaults\n";
            exit(1);
        }

        args.indir = vm["input"].as<string>();
        args.mapdir = vm["map"].as<string>();
    }
}

/**
 * Configure & assign output directory
 * @param vm Command-line variable map
 * @param args Parallel-PMF program arguments
 */
void configureOutput(po::variables_map &vm, Arguments &args)
{
    if (!vm.count("output"))
    {
        args.outdir = "results";
    }
    else
    {
        args.outdir = vm["output"].as<string>();
    }

    if (filesystem::exists(args.outdir))
    {
        cout << "Outdir " << args.outdir << " exists" << endl;
    }
    else
    {
        cout << "Outdir doesn't exist, creating " << args.outdir << "..." << endl;
        filesystem::create_directory(args.outdir);
    }
}

/**
 * Configure & assign model task
 * @param vm Command-line variable map
 * @param args Parallel-PMF program arguments
 */
void configureTask(po::variables_map &vm, Arguments &args)
{
    if (vm["task"].as<string>() == "train")
    {
        args.task = "train";
    }
    else if (vm["task"].as<string>() == "recommend")
    {
        // "user" & "item" options -- XOR with each other
        Expects(!vm["user"].as<bool>() != !vm["item"].as<bool>());

        args.task = "recommend";
        args.rec_option = (vm["user"].as<bool>()) ? RecOption::user : RecOption::item;
    }
    else
    {
        cout << "Unsupported recommendation option, please choose one from either `--user` or `--item`" << endl;
        exit(1);
    }
}

/**
 * Configure & assign model specification variables
 * @param vm Command-line variable map
 * @param args Parallel-PMF program arguments
 */
void configurePMF(po::variables_map &vm, Arguments &args)
{
    args.n_threads = vm["thread"].as<int>();
    args.run_fit_sequential = vm["run_sequential"].as<bool>();

    args.k = vm["n_components"].as<int>();
    args.n_epochs = vm["n_epochs"].as<int>();
    args.gamma = vm["gamma"].as<double>();
    args.ratio = vm["ratio"].as<double>();
    args.std_theta = vm["std_theta"].as<double>();
    args.std_beta = vm["std_beta"].as<double>();
    args.loss_interval = vm["loss_interval"].as<int>();

    if (args.n_threads < 2)
    {
        cerr << "Num threads must be at least 2. User provided: " << args.n_threads << endl;
        exit(1);
    }
}

/**
 * Tokenize (split) a string into vector of compoments delimited by the same separater (e.g. comma, tab, etc.)
 * @param str Input string to be splitted
 * @param delimiter Separator character
 * @return Vector of tokenized strings
 */
vector<string> tokenize(string &str, const string delimiter)
{
    vector<string> tokenized{};
    boost::split(tokenized, str, boost::is_any_of(delimiter), boost::token_compress_on);

    return tokenized;
}

/**
 * Return vector of non-negative indices from the given vector
 * @param x Vector of doubles
 * @return Indices of non-negative indices of the given vector
 */
vector<int> nonNegativeIdxs(const VectorXd &x)
{
    vector<int> indices{};
    for (int i = 0; i < x.size(); i++)
    {
        if (x[i] >= 0)
        {
            indices.push_back(i);
        }
    }

    return indices;
}

/**
 * Count number of intersect items between two input vectors
 * @param x Vector of integers (Eigen object)
 * @param y Vector of integers (Eigen object)
 * @return Number of common items between vectors x & y
 */
int countIntersect(const VectorXi &x, const VectorXi &y)
{
    vector<int> vi_x(x.size());
    vector<int> vi_y(y.size());
    VectorXi::Map(vi_x.data(), x.size()) = x;
    VectorXi::Map(vi_y.data(), y.size()) = y;

    vector<int> intersect{};
    set_intersection(vi_x.begin(), vi_x.end(), vi_y.begin(), vi_y.end(), back_inserter(intersect));

    return intersect.size();
}

/**
 * Return the indices that would sort the input vector
 * @param x Input vector of doubles (Eigen object)
 * @param option Sorting option: (Order::ascend or Order::descend)
 * @return Vector of indices represent the sorted order of the input vector
 */
VectorXi argsort(const VectorXd &x, const Order option)
{
    Expects(option == Order::ascend or option == Order::descend);

    vector<double> vi(x.size());
    VectorXd::Map(vi.data(), x.size()) = x;

    vector<int> indices(x.size());
    int idx = 0;
    std::generate(indices.begin(), indices.end(), [&] { return idx++; });

    if (option == Order::ascend)
    {
        std::sort(indices.begin(), indices.end(), [&](int a, int b) { return vi[a] < vi[b]; });
    }
    else
    {
        std::sort(indices.begin(), indices.end(), [&](int a, int b) { return vi[a] > vi[b]; });
    }

    Eigen::Map<VectorXi> indices_sorted(indices.data(), indices.size());

    return indices_sorted;
}

/**
 * Return the unique items in a given column of the input matrix
 * @param mat Pointer of the input matrix
 * @param col_idx Column index of the matrix to calculate unique items
 * @return vector of unique items from the given column of the input matrix
 */
vector<int> getUnique(const shared_ptr<MatrixXd> &mat, int col_idx)
{
    Expects(col_idx >= 0 and col_idx <= mat->cols());

    const MatrixXd &col = mat->col(col_idx);
    set<int> unique_set{col.data(), col.data() + col.size()};
    vector<int> unique(unique_set.begin(), unique_set.end());

    return unique;
}

/**
 * Calculate root-mean-squared error (RMSE) between a vector and an integer representing the constant prediction
 * @param y Ground-truth vector
 * @param y_hat Integer that represent a vector of constant predictions
 * @return sqrt( ∑_i(y_i - y_hat_i)^2 / size(y))
 */
double rmse(const VectorXd &y, const double y_hat)
{
    const auto denominator = y.size();
    Expects(denominator > 0);

    const double c = 1 / (double)denominator;
    const VectorXd &squared = (y.array() - y_hat).square();
    const double &summed = squared.sum();

    return sqrt(c * summed);
}

/**
 * Calculate root-mean-squared error (RMSE) between two vectors
 * @param y Ground-truth vector
 * @param y_hat Prediction vector
 * @return sqrt( ∑_i(y_i - y_hat_i)^2 / size(y))
 */
double rmse(const VectorXd &y, const VectorXd &y_hat)
{
    Expects(y.size() > 0);
    Expects(y.size() == y_hat.size());

    const VectorXd &err = y - y_hat;
    const VectorXd &sq_err = err.array().square();
    const double &ms_error = sq_err.sum() / y.size();

    return sqrt(ms_error);
}

/**
 * Calculate Coefficient of determination between two vectors
 * @param y Ground-truth vector
 * @param y_hat Prediction vector
 * @return  Coefficient of determination between y & y_hat
 */
double r2(const VectorXd &y, const VectorXd &y_hat)
{
    Expects(y.size() > 0);
    Expects(y.size() == y_hat.size());

    VectorXd y_mean(y.size());
    y_mean.setConstant(y.mean());
    double SSE = (y - y_hat).array().square().sum();
    double TSS = (y - y_mean).array().square().sum();
    return 1 - (SSE / TSS);
}

/**
 * Calculate cosine similarity between two vectors
 * @param v1 Vector 1
 * @param v2 Vector 2
 * @return cosine(y1, y2) = (y1 dot y2) / (||y1|| * ||y2||)
 */
double cosine(const VectorXd &v1, const VectorXd &v2)
{
    Expects(v1.size() > 0);
    Expects(v1.size() == v2.size());
    double distance = v1.dot(v2) / (v1.norm() * v2.norm());

    return distance;
}

} // namespace Utils