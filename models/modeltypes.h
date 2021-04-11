#ifndef FINAL_PROJECT_MODELTYPES_H
#define FINAL_PROJECT_MODELTYPES_H

#include <Eigen/Dense>
#include <map>
#include <memory>

namespace Model
{
using namespace std;
using namespace Eigen;

using LatentVectors = map<int, VectorXd>;
using TrainingData = shared_ptr<MatrixXd>;
using TestingData = shared_ptr<MatrixXd>;

} // namespace Model

#endif // FINAL_PROJECT_MODELTYPES_H
