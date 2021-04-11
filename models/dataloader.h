#ifndef FINAL_PROJECT_DATALOADER_H
#define FINAL_PROJECT_DATALOADER_H

#include "../csvlib/csv.h"
#include "abstractdataloader.h"
#include "modeltypes.h"
#include <Eigen/Dense>
#include <filesystem>
namespace Model
{

using namespace std;
using namespace Eigen;

class DataLoader : public AbstractDataLoader
{
  private:
    const filesystem::path m_dataset_in;
    const filesystem::path m_res_dir;

    MatrixXd loadFromFile();

  public:
    DataLoader(const string &dataset_in, const string &m_res_dir);

    virtual MatrixXd getDataset() const;

    virtual tuple<LatentVectors, LatentVectors> getLearntVectors() const;

    virtual void saveTrainResults(const LatentVectors &theta, const LatentVectors &beta,
                                  const vector<double> &losses) const;
};

} // namespace Model

#endif // FINAL_PROJECT_DATALOADER_H
