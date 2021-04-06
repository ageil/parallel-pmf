#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>

#include "models/DataManager.h"
#include "models/PMF.h"
#include "models/utils.h"

#include <boost/program_options.hpp>

using namespace std;
using namespace Model;
using namespace Utils;
using namespace chrono;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    // Initialize default values for arguments, path configuration
    string task = "train";
    Model::RecOption rec_option;

    string input = "./movielens/ratings.csv";
    string map_input = "./movielens/movies.csv";
    filesystem::path outdir("results");

    int k = 3;
    int n_epochs = 200;  // default # of iterations
    double gamma = 0.01; // default learning rate for gradient descent
    double ratio = 0.7;  // train-test split ratio
    int n_threads = 2;

    double std_theta = 1.0;
    double std_beta = 1.0;

    po::options_description desc("Parameters for Probabilistic Matrix Factorization (PMF)");
    desc.add_options()("help,h", "Help")("input,i", po::value<string>(&input), "Input file name")(
        "map,m", po::value<string>(&map_input), "Item mapping file name")(
        "task", po::value<string>(&task), "Task to perform\n [Options: 'train', 'recommend']")(
        "output,o", po::value<filesystem::path>(&outdir), "Output directory\n  [default: current_path/results/]")(
        "n_components,k", po::value<int>(&k), "Number of components (k)\n [default: 3]")(
        "n_epochs,n", po::value<int>(&n_epochs), "Num. of learning iterations\n  [default: 200]")(
        "ratio,r", po::value<double>(&ratio), "Ratio for training/test set splitting\n [default: 0.7]")(
        "thread", po::value<int>(&n_threads), "Number of threads for parallelization")(
        "gamma", po::value<double>(&gamma), "Learning rate for gradient descent\n  [default: 0.01]")(
        "std_theta", po::value<double>(&std_theta), "Std. of theta's prior normal distribution\n  [default: 1]")(
        "std_beta", po::value<double>(&std_beta), "Std. of beta's prior normal distribution\n  [default: 1]")(
        "user", po::bool_switch()->default_value(false), "Recommend items for given user")(
        "item", po::bool_switch()->default_value(false), "Recommend similar items for a given item")(
        "genre", po::bool_switch()->default_value(false), "Recommend items for a given genre");

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

    if (vm["user"].as<bool>())
    {
        rec_option = Model::RecOption::user;
    }
    else if (vm["item"].as<bool>())
    {
        rec_option = Model::RecOption::item;
    }
    else if (vm["genre"].as<bool>())
    {
        rec_option = Model::RecOption::genre;
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

    if (task == "train")
    {
        // (1). Load datasets, split to training & test sets
        Utils::DataManager dm = DataManager();
        shared_ptr<MatrixXd> ratings = dm.load(input, ratio); // load main dataset
        // shared_ptr<MatrixXd> ratings_train = dm.getTrain();
        // shared_ptr<MatrixXd> ratings_test = dm.getTest();

        // (2). Initialize Model object
        Model::PMF model{ratings, k, std_beta, std_theta, dm.users, dm.items};

        // (3). Train: fit the model to the training data
        auto t0 = chrono::steady_clock::now();
        vector<double> losses = model.fit(n_epochs, gamma, n_threads);
        auto t1 = chrono::steady_clock::now();
        double delta_t = std::chrono::duration<double, std::milli>(t1 - t0).count() * 0.001;
        cout << "Running time for " << n_epochs << " iterations: " << delta_t << " s." << endl;
        cout << endl;

        // (3.1) Evaluate model quality on test data
        VectorXd actual = ratings->rightCols(1);
        VectorXd predicted = model.predict(ratings->leftCols(2));
        double error = Utils::rmse(actual, predicted);
        double baseline_zero = Utils::rmse(actual, 0.0);
        double baseline_avg = Utils::rmse(actual, actual.mean());

        cout << "RMSE(0): " << baseline_zero << endl;
        cout << "RMSE(mean): " << baseline_avg << endl;
        cout << "RMSE(pred): " << error << endl;

        // precision & recall of the top N items recommended for each user [Not in use]
        /*
        vector<int> vec_N {10, 50, 100, 500};
        for (auto &N : vec_N)
        {
            Metrics acc = model.accuracy(ratings_train, N);
            cout << "Metrics(pred) for top " << N << " recommended items for each user\n"
                 << "Precision: " << acc.precision << " Recall: " << acc.recall << endl;
        }
         */

        // (3.2) save loss & trained parameters to file
        model.save(outdir);
    }
    else if (task == "recommend")
    {
        // (1). Load datasets, split to training & test sets
        Utils::DataManager dm = DataManager();
        shared_ptr<MatrixXd> ratings = dm.load(input, ratio); // load main dataset

        // (2). Initialize Model object
        Model::PMF model{ratings, k, std_beta, std_theta, dm.users, dm.items};

        // (3). Recommendatations
        // (3.1) Load model from file
        model.load(outdir);
        ItemMap item_map = dm.loadItemMap(map_input);

        if (rec_option == RecOption::user)
        {
            // (3.2-1) recommend user
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
                cout << "\nTop 10 recommended movies for user " << user_id << " :" << endl << endl;
                for (auto &title : rec)
                {
                    cout << "Movie: " << title << '\t' << "Genre: " << item_map.name_genre[title] << endl;
                }
            }
        }
        else if (rec_option == RecOption::item)
        {
            // (3.2-2) recommend similar items
            cout << "Please specify movie name: " << endl;
            string item_name;
            getline(cin, item_name);

            if (item_map.name_id.find(item_name) == item_map.name_id.end())
            {
                cerr << "Movie " << item_name << " doesn't exist in the dataset" << endl;
            }
            else
            {
                vector<string> rec = model.getSimilarItems(item_map.name_id[item_name], item_map.id_name, 10);
                cout << "\nTop 10 similar movies to " << item_name << " :" << endl << endl;
                for (auto &title : rec)
                {
                    cout << "Movie: " << title << '\t' << "Genre: " << item_map.name_genre[title] << endl;
                }
            }
        }
        else
        {
            // (3.2-3) recommend from genre
            cout << "Please specify genre: " << endl;
            string genre;
            getline(cin, genre);

            if (item_map.genre_ids.find(genre) == item_map.genre_ids.end())
            {
                cerr << "Genre " << genre << " doesn't exist in the dataset" << endl;
            }
            else
            {
                vector<string> rec = model.recommendByGenre(genre, item_map.id_name, item_map.genre_ids, 10);
                cout << "\n10 recommended movies for genre " << genre << " :" << endl << endl;
                for (auto &title : rec)
                {
                    cout << "Movie: " << title << '\t' << "Genre: " << item_map.name_genre[title] << endl;
                }
            }
        }
    }

    return 0;
}
