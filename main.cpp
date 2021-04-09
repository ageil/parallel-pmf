#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

#include "models/PMF.h"
#include "models/datamanager.h"
#include "models/utils.h"

#include <boost/program_options.hpp>

using namespace std;
using namespace Model;
using namespace Utils;
using namespace chrono;
namespace po = boost::program_options;

namespace
{
/**
 * Enums of the possible user options for recomendations
 */
enum class RecOption
{
    user = 0,
    item = 1
};

} // namespace

int main(int argc, char **argv)
{
    // Initialize default values for arguments, path configuration
    string task = "train";
    RecOption rec_option;

    string input = "";
    string map_input = "";
    filesystem::path outdir("results");

    int k = 3;
    int n_epochs = 200;  // default # of iterations
    double gamma = 0.01; // default learning rate for gradient descent
    double ratio = 0.7;  // train-test split ratio
    int n_threads = 20;
    double std_theta = 1.0;
    double std_beta = 1.0;
    int loss_interval = 10;

    // clang-format off

    po::options_description desc("Parameters for Probabilistic Matrix Factorization (PMF)");
    
    desc.add_options()
        ("help,h", "Help")
        ("input,i", po::value<string>(&input), "Input file name")
        ("map,m", po::value<string>(&map_input), "Item mapping file name")
        ("use_defaults,d", po::bool_switch()->default_value(false), "If enabled, uses './movielens/ratings.csv' for the input file and './movielens/movies.csv' for the map input file")
        ("output,o", po::value<filesystem::path>(&outdir), "Output directory\n[default: current_path/results/]\n\n")
        ("task", po::value<string>(&task), "Task to perform\n[Options: 'train', 'recommend']\n")
        ("n_components,k", po::value<int>(&k), "Number of components (k)\n[default: 3]\n")
        ("n_epochs,n", po::value<int>(&n_epochs), "Num. of learning iterations\n[default: 200]\n")
        ("ratio,r", po::value<double>(&ratio), "Ratio for training/test set splitting\n[default: 0.7]\n")
        ("thread", po::value<int>(&n_threads), "Number of threads for parallelization\nThis value must be at least 2\n")
        ("gamma", po::value<double>(&gamma), "Learning rate for gradient descent\n[default: 0.01]\n")
        ("std_theta", po::value<double>(&std_theta), "Std. of theta's prior normal distribution\n[default: 1]\n")
        ("std_beta", po::value<double>(&std_beta), "Std. of beta's prior normal distribution\n[default: 1]\n")
        ("run_sequential,s", po::bool_switch()->default_value(false), "Enable running model fitting sequentially\n")
        ("user", po::bool_switch()->default_value(false), "Recommend items for given user\n")
        ("item", po::bool_switch()->default_value(false), "Recommend similar items for a given item\n")
        ("loss_interval, l", po::value<int>(&loss_interval), "Number of epochs between each loss computation.\n[default: 10]");

    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        cout << desc << endl;
        return 0;
    }

    if (n_threads < 2)
    {
        cout << "Num threads must be at least 2. User provided: " << n_threads << endl;
        return 1;
    }

    if (outdir.empty())
    {
        outdir = "results";
    }

    if (vm["use_defaults"].as<bool>())
    {
        input = "./movielens/ratings.csv";
        map_input = "./movielens/movies.csv";
        cout << "[ use_defaults, d ] enabled. Setting input as " << input << " and map_input as " << map_input << endl;
    }
    else
    {
        if (input.empty())
        {
            cout << "[ input, i ] not provided. Please provide an input file or specify -d to use defaults\n";
            return 1;
        }

        if (map_input.empty())
        {
            cout << "[ map, m ] not provided. Please provide an map input file or specify -d to use defaults\n";
            return 1;
        }
    }

    if (vm["user"].as<bool>())
    {
        rec_option = RecOption::user;
    }
    else if (vm["item"].as<bool>())
    {
        rec_option = RecOption::item;
    }

    if (filesystem::exists(outdir))
    {
        cout << "Outdir " << outdir << " exists" << endl;
    }
    else
    {
        cout << "Outdir doesn't exist, creating " << outdir << "..." << endl;
        filesystem::create_directory(outdir);
    }

    const bool run_fit_sequential = vm["run_sequential"].as<bool>();

    // (1). read CSV & split to training & test sets
    auto dm_t0 = chrono::steady_clock::now();

    const auto data_mgr = make_shared<DataManager::DataManager>(input, ratio);

    auto dm_t1 = chrono::steady_clock::now();
    double dm_delta_t = std::chrono::duration<double, std::milli>(dm_t1 - dm_t0).count() * 0.001;
    cout << "Took " << dm_delta_t << " s. to load data. \n\n";

    // (2). Fit the model to the training data
    Model::PMF model{data_mgr, k, std_beta, std_theta, loss_interval};

    if (task == "train")
    {
        auto fit_t0 = chrono::steady_clock::now();
        vector<double> losses;

        if (run_fit_sequential or n_threads == 1)
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

        // (3.1) Evaluate model quality on test data
        shared_ptr<MatrixXd> ratings_test = data_mgr->getTest();
        VectorXd actual = ratings_test->rightCols(1);
        VectorXd predicted = model.predict(ratings_test->leftCols(2));

        double error = Utils::rmse(actual, predicted);
        double baseline_zero = Utils::rmse(actual, 0.0);
        double baseline_avg = Utils::rmse(actual, actual.mean());

        cout << "RMSE(0): " << baseline_zero << endl;
        cout << "RMSE(mean): " << baseline_avg << endl;
        cout << "RMSE(pred): " << error << endl;

        // (3.2) save loss & trained parameters to file
        model.save(outdir);
    }
    else if (task == "recommend")
    {
        // (3). Recommendations
        // (3.1) Load model from file
        model.load(outdir);

        DataManager::ItemMap item_map = data_mgr->loadItemMap(map_input);

        if (rec_option == RecOption::user)
        {
            // (3.2-1) recommend to user
            cout << "Please specify user id: " << endl;
            int user_id;
            cin >> user_id;

            if (item_map.id_name.find(user_id) == item_map.id_name.end())
            {
                cerr << "User id " << user_id << " doesn't exist in the dataset" << endl;
            }
            else
            {
                vector<string> rec = model.recommend(user_id, item_map.id_name, 10);
                cout << "\nTop 10 recommended items for user " << user_id << " :" << endl << endl;
                for (auto &item : rec)
                {
                    cout << "Item: " << item << '\t' << "Attribute: " << item_map.name_item_attributes[item] << endl;
                }
            }
        }
        else if (rec_option == RecOption::item)
        {
            // (3.2-2) recommend similar items
            cout << "Please specify item name: " << endl;
            string item_name;
            getline(cin, item_name);

            if (item_map.name_id.find(item_name) == item_map.name_id.end())
            {
                cerr << "Item " << item_name << " doesn't exist in the dataset" << endl;
            }
            else
            {
                vector<string> rec = model.getSimilarItems(item_map.name_id[item_name], item_map.id_name, 10);
                cout << "\nTop 10 similar items to " << item_name << " :" << endl << endl;
                for (auto &item : rec)
                {
                    cout << "Item: " << item << '\t' << "Attributes: " << item_map.name_item_attributes[item] << endl;
                }
            }
        }
        else
        {
            cout << "Unsuppored recommendation option. \n";
            return 1;
        }
    }

    return 0;
}
