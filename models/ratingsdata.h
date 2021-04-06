#ifndef FINAL_PROJECT_RATINGSDATA_H
#define FINAL_PROJECT_RATINGSDATA_H

namespace Model
{
namespace RatingsData
{

enum class Cols
{
    user = 0,
    item = 1,
    rating = 2
};

int col_value(Cols);

} // namespace RatingsData
} // namespace Model

#endif