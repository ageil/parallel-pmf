#include "PMF.h"

PMF::PMF(MatrixXd _data, int _k, double _std_theta, double _std_beta) {
    data = _data;
    k = _k;
    std_beta = _std_beta;
    std_theta = _std_theta;
    losses.clear();

    default_random_engine generator(time(nullptr));
    normal_distribution<double> dist_beta(0, std_beta);
    normal_distribution<double> dist_theta(0, std_theta);

    for (int i = 0; i < k; i++) {
        beta.push_back(dist_beta(generator));
        theta.push_back(dist_theta(generator));
    }
}

double PMF::normPDF(int x, double loc, double scale) {
    cerr << "Not implemented yet" << endl;
    return 0;
}

double PMF::logNormPDF(int x, double loc, double scale) {
    cerr << "Not implemented yet" << endl;
    return 0;
}

double PMF::gradLogNormPDF(int x, double loc, double scale) {
    cerr << "Not implemented yet" << endl;
    return 0;
}

vector<double> PMF::loss(MatrixXd _data) {
    cerr << "Not implemented yet" << endl;
    vector<double> tmp {};
    return tmp;
}

MatrixXd PMF::predict(MatrixXd _data) {
    cerr << "Not implemented yet" << endl;
    MatrixXd m(1,1);
    return m;
}

VectorXd PMF::recommend(int user) {
    cerr << "Not implemented yet" << endl;
    VectorXd v;
    return v;
}
