#ifndef FINAL_PROJECT_PMF_H
#define FINAL_PROJECT_PMF_H

#include <iostream>
#include <Eigen/Dense>
#include <random>
using namespace std;
using namespace Eigen;

class PMF {
private:
    double k;
    double std_theta;
    double std_beta;
    vector<double> beta {};
    vector<double> theta {};
    vector<double> losses {};
    MatrixXd data;

public:
    PMF(MatrixXd d, int n_components, double eta_theta, double eta_beta);
    ~PMF();
    double normPDF(int x, double loc=0.0, double scale=1.0);
    double logNormPDF(int x, double loc=0.0, double scale=1.0);
    double gradLogNormPDF(int x, double loc=0.0, double scale=1.0);
    vector<double> loss(MatrixXd _data);
    MatrixXd predict(MatrixXd _data);
    VectorXd recommend(int user);

    //todo: specify return type for "norm_vectors" functions in model.py
};
