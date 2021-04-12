#ifndef FINAL_PROJECT_ABSTRACTDATALOADER_H
#define FINAL_PROJECT_ABSTRACTDATALOADER_H

#include "modeltypes.h"
#include <Eigen/Dense>
#include <memory>
#include <tuple>

namespace Model
{
using namespace std;
using namespace Eigen;

/**
 * Barebone interface for a `DataLoader`, agnostic of what the source of the data is (e.g. database, csv files on disk
 * etc.)
 */
class AbstractDataLoader
{
  public:
    virtual MatrixXd getDataset() const = 0;

    virtual tuple<LatentVectors, LatentVectors> getLearntVectors() const = 0;

    virtual void saveTrainResults(const LatentVectors &theta, const LatentVectors &beta,
                                  const vector<double> &losses) const = 0;
};

} // namespace Model

#endif // FINAL_PROJECT_ABSTRACTDATALOADER_H
