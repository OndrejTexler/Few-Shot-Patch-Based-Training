// Copyright(C) 2018 Tommy Hinks <tommy.hinks@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#ifndef THINKS_POISSON_DISK_SAMPLING_POISSON_DISK_SAMPLING_H_INCLUDED
#define THINKS_POISSON_DISK_SAMPLING_POISSON_DISK_SAMPLING_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

namespace thinks {
namespace poisson_disk_sampling {
namespace detail {

namespace util {

/*!
Returns value clamped to the min/max boundaries.
*/
template <typename T>
T clamp(const T min_value, const T max_value, const T value) {
  assert(min_value <= max_value && "min_value <= max_value");
  return value < min_value ? min_value
                           : (value > max_value ? max_value : value);
}

/*!
Returns x squared (not checking for overflow).
*/
template <typename T>
T squared(const T x) {
  return x * x;
}

/*!
Returns the squared magnitude of a.
*/
template <typename FloatT, std::size_t N>
typename std::array<FloatT, N>::value_type SquaredMagnitude(
    const std::array<FloatT, N>& a) {
  constexpr auto kDims = std::tuple_size<std::array<FloatT, N>>::value;
  static_assert(kDims >= 1, "dimensions must be >= 1");

  auto m = squared(a[0]);
  for (auto i = std::size_t{1}; i < kDims; ++i) {
    m += squared(a[i]);
  }
  return m;
}

/*!
Returns the squared distance between the vectors u and v.
*/
template <typename VecTraitsT, typename VecT>
typename VecTraitsT::ValueType SquaredDistance(const VecT& u, const VecT& v) {
  static_assert(VecTraitsT::kSize >= 1, "dimensions must be >= 1");

  auto d = squared(VecTraitsT::Get(u, 0) - VecTraitsT::Get(v, 0));
  for (auto i = std::size_t{1}; i < VecTraitsT::kSize; ++i) {
    d += squared(VecTraitsT::Get(u, i) - VecTraitsT::Get(v, i));
  }
  return d;
}

/*!
Returns true if x is element-wise inside x_min and x_max, otherwise false.

Note: The bounds checking is inclusive, such that x == x_min or x == x_max
return true.
*/
template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
bool Inside(const VecT& x, const std::array<FloatT, N>& x_min,
            const std::array<FloatT, N>& x_max) {
  constexpr auto kDims = std::tuple_size<std::array<FloatT, N>>::value;
  static_assert(VecTraitsT::kSize == kDims, "dimensionality mismatch");

  for (auto i = std::size_t{0}; i < VecTraitsT::kSize; ++i) {
    const auto xi = static_cast<FloatT>(VecTraitsT::Get(x, i));
    assert(x_min[i] < x_max[i] && "min < max");
    if (x_min[i] > xi || xi > x_max[i]) {
      return false;
    }
  }
  return true;
}

/*!
Erase the value at index in the vector v. The vector v is
guaranteed to decrese in size by one. Note that the ordering of elements
in v may change as a result of calling this function.
Assumes that v is non-null and not empty.
*/
template <typename T>
void EraseUnordered(std::vector<T>* const v, const std::size_t index) {
  assert(v != nullptr);
  assert(!v->empty());
  assert(index < v->size());

  (*v)[index] = v->back();
  v->pop_back();  // O(1).
}

/*!
Increment index such that repeated calls to this function enumerate
all iterations between min_index and max_index (inclusive).
*/
template <typename IntT, std::size_t N>
bool Iterate(const std::array<IntT, N>& min_index,
             const std::array<IntT, N>& max_index,
             std::array<IntT, N>* const index) {
  constexpr auto kDims = std::tuple_size<std::array<IntT, N>>::value;
  static_assert(kDims >= 1, "dimensions must be >= 1");

  auto i = std::size_t{0};
  for (; i < kDims; ++i) {
    assert(min_index[i] <= max_index[i] && "min <= max");
    assert(min_index[i] <= (*index)[i] && (*index)[i] <= max_index[i] &&
           "min <= index <= max");

    (*index)[i]++;
    if ((*index)[i] <= max_index[i]) {
      break;
    }
    (*index)[i] = min_index[i];
  }
  return i == kDims ? false : true;
}

}  // namespace util

namespace rand {

/*!
Stateless and repeatable function that returns a
pseduo-random number in the range [0, 0xFFFFFFFF].
*/
inline std::uint32_t Hash(const std::uint32_t seed) {
  // So that we can use unsigned int literals, e.g. 42u.
  static_assert(sizeof(unsigned int) == sizeof(std::uint32_t),
                "integer size mismatch");

  auto i = std::uint32_t{(seed ^ 12345391u) * 2654435769u};
  i ^= (i << 6u) ^ (i >> 26u);
  i *= 2654435769u;
  i += (i << 5u) ^ (i >> 12u);
  return i;
}

/*!
Returns a pseduo-random number in the range [0, 1].
*/
template <typename FloatT>
FloatT NormRand(const std::uint32_t seed) {
  static_assert(std::is_floating_point<FloatT>::value,
                "FloatT must be floating point");

  constexpr auto scale =
      FloatT{1} /
      static_cast<FloatT>(std::numeric_limits<std::uint32_t>::max());

  return scale * static_cast<FloatT>(Hash(seed));
}

/*!
Returns a pseduo-random number in the range [offset, offset + range].
*/
template <typename FloatT>
FloatT RangeRand(const FloatT offset, const FloatT range,
                 const std::uint32_t seed) {
  return offset + range * NormRand<FloatT>(seed);
}

/*!
Returns an array where each element has been assigned using RangeRand,
with bounds taken from the corresponding element in x_min and x_max.
Note that seed is incremented for each assigned element.
*/
template <typename FloatT, std::size_t N>
std::array<FloatT, N> ArrayRangeRand(const std::array<FloatT, N>& x_min,
                                     const std::array<FloatT, N>& x_max,
                                     std::uint32_t* const seed) {
  constexpr auto kDims = std::tuple_size<std::array<FloatT, N>>::value;

  assert(seed != nullptr);

  auto a = std::array<FloatT, N>{};
  for (auto i = std::size_t{0}; i < kDims; ++i) {
    assert(x_min[i] < x_max[i] && "min < max");

    const auto offset = x_min[i];
    const auto range = x_max[i] - x_min[i];

    // Not worrying about seed "overflow" since it is unsigned.
    a[i] = RangeRand(offset, range, (*seed)++);
  }
  return a;
}

/*!
See ArrayRangeRand.
*/
template <typename VecT, typename VecTraitsT, typename FloatT, std::size_t N>
VecT VecRangeRand(const std::array<FloatT, N>& x_min,
                  const std::array<FloatT, N>& x_max,
                  std::uint32_t* const seed) {
  constexpr auto kDims = std::tuple_size<std::array<FloatT, N>>::value;
  static_assert(VecTraitsT::kSize == kDims, "dimensionality mismatch");

  auto v = VecT{};
  const auto a = ArrayRangeRand(x_min, x_max, seed);
  for (auto i = std::size_t{0}; i < kDims; ++i) {
    VecTraitsT::Set(&v, i, static_cast<typename VecTraitsT::ValueType>(a[i]));
  }
  return v;
}

/*!
Returns a pseudo-random index in the range [0, size - 1].
*/
inline std::size_t IndexRand(const std::size_t size,
                             std::uint32_t* const seed) {
  return static_cast<std::size_t>(
      RangeRand(float{0}, static_cast<float>(size) - static_cast<float>(0.0001),
                (*seed)++));
}

}  // namespace rand

namespace grid {

template <typename FloatT, std::size_t N>
class Grid {
 public:
  typedef std::int32_t CellType;
  typedef std::array<std::int32_t, N> IndexType;

  static constexpr auto kDims = std::tuple_size<IndexType>::value;

  static_assert(kDims >= 1, "grid dimensionality must be >= 1");
  static_assert(std::is_floating_point<FloatT>::value,
                "FloatT must be floating point");

  Grid(const FloatT sample_radius, const std::array<FloatT, N>& x_min,
       const std::array<FloatT, N>& x_max)
      : sample_radius_(sample_radius),
        dx_(GetDx_(sample_radius_)),
        dx_inv_(FloatT{1} / dx_),
        x_min_(x_min),
        x_max_(x_max),
        size_(GetGridSize_(x_min_, x_max_, dx_inv_)),
        cells_(GetCells_(size_)) {}

  FloatT sample_radius() const { return sample_radius_; }

  IndexType size() const { return size_; }

  /*!
  Returns the index for a position along the i'th axis.
  Note that the returned index may be negative.
  */
  template <typename FloatT2>
  typename IndexType::value_type AxisIndex(const std::size_t i,
                                           const FloatT2 pos) const {
    typedef typename IndexType::value_type IndexValueType;

    return static_cast<IndexValueType>((static_cast<FloatT>(pos) - x_min_[i]) *
                                       dx_inv_);
  }

  /*!
  Note that the returned index elements may be negative.
  */
  template <typename VecTraitsT, typename VecT>
  IndexType IndexFromSample(const VecT& sample) const {
    static_assert(VecTraitsT::kSize == kDims, "dimensionality mismatch");

    auto index = IndexType{};
    for (auto i = std::size_t{0}; i < kDims; ++i) {
      index[i] = AxisIndex(i, VecTraitsT::Get(sample, i));
    }
    return index;
  }

  CellType Cell(const IndexType& index) const {
    return cells_[LinearIndex(index)];
  }

  CellType& Cell(const IndexType& index) { return cells_[LinearIndex(index)]; }

 private:
  FloatT sample_radius_;
  FloatT dx_;
  FloatT dx_inv_;
  std::array<FloatT, N> x_min_;
  std::array<FloatT, N> x_max_;
  IndexType size_;
  std::vector<CellType> cells_;

  std::size_t LinearIndex(const IndexType& index) const {
    assert(index[0] >= 0 && "negative index");
    assert(index[0] < size_[0] && "index outside grid");
    auto k = static_cast<std::size_t>(index[0]);
    auto d = std::size_t{1};
    for (auto i = std::size_t{1}; i < kDims; ++i) {
      assert(index[i] >= 0 && "negative index");
      assert(index[i] < size_[i] && "index outside grid");

      // Note: Not checking for "overflow".
      d *= static_cast<std::size_t>(size_[i - 1]);
      k += static_cast<std::size_t>(index[i]) * d;
    }
    return k;
  }

  static FloatT GetDx_(const FloatT sample_radius) {
    assert(sample_radius > FloatT{0} && "sample_radius > 0");

    // The grid cell size should be such that each cell can only
    // contain one sample. We apply a scaling factor to avoid
    // numerical issues.
    constexpr auto eps = static_cast<FloatT>(0.001);
    constexpr auto scale = FloatT{1} - eps;
    return scale * sample_radius / std::sqrt(static_cast<FloatT>(N));
  }

  static IndexType GetGridSize_(const std::array<FloatT, N>& x_min,
                                const std::array<FloatT, N>& x_max,
                                const FloatT dx_inv) {
    assert(dx_inv > FloatT{0} && "1/dx > 0");

    // Compute size in each dimension using grid cell size (dx).
    auto grid_size = IndexType{};
    for (auto i = std::size_t{0}; i < kDims; ++i) {
      assert(x_min[i] < x_max[i] && "min < max");

      grid_size[i] = static_cast<typename IndexType::value_type>(
          std::ceil((x_max[i] - x_min[i]) * dx_inv));

      assert(grid_size[i] >= 1 && "grid_size >= 1");
    }
    return grid_size;
  }

  static std::vector<std::int32_t> GetCells_(const IndexType& size) {
    // Initialize cells with value -1, indicating no sample there.
    // Cell values are later set to indices of samples.
    const auto linear_size =
        std::accumulate(std::begin(size), std::end(size), std::size_t{1},
                        std::multiplies<std::size_t>());
    return std::vector<CellType>(linear_size, -1);
  }
};

/*!
Named contructor that helps with type deduction.
*/
template <typename FloatT, std::size_t N>
Grid<FloatT, N> MakeGrid(const FloatT sample_radius,
                         const std::array<FloatT, N>& x_min,
                         const std::array<FloatT, N>& x_max) {
  return Grid<FloatT, N>(sample_radius, x_min, x_max);
}

}  // namespace grid

template <typename FloatT>
void ThrowIfInvalidRadius(const FloatT radius) {
  if (!(radius > FloatT{0})) {
    std::ostringstream oss{};
    oss << "radius must be positive, was " << radius;
    throw std::invalid_argument(oss.str());
  }
}

template <typename FloatT, std::size_t N>
void ThrowIfInvalidBounds(const std::array<FloatT, N>& x_min,
                          const std::array<FloatT, N>& x_max) {
  constexpr auto kDims = std::tuple_size<std::array<FloatT, N>>::value;
  static_assert(kDims >= 1, "bounds dimensionality must be >= 1");

  auto i = std::size_t{0};
  for (; i < kDims; ++i) {
    if (!(x_max[i] > x_min[i])) {
      break;
    }
  }

  if (i < kDims) {
    std::ostringstream oss_min{};
    std::ostringstream oss_max{};
    oss_min << "[";
    oss_max << "[";
    for (auto j = std::size_t{0}; j < kDims; ++j) {
      oss_min << x_min[j] << (j != kDims - 1 ? ", " : "");
      oss_max << x_max[j] << (j != kDims - 1 ? ", " : "");
    }
    oss_min << "]";
    oss_max << "]";

    std::ostringstream oss{};
    oss << "invalid bounds - max must be greater than min, was "
        << "min: " << oss_min.str() << ", "
        << "max: " << oss_max.str();

    throw std::invalid_argument(oss.str());
  }
}

inline void ThrowIfInvalidMaxSampleAttempts(
    const std::uint32_t max_sample_attempts) {
  if (!(max_sample_attempts > 0)) {
    std::ostringstream oss{};
    oss << "max sample attempts must be greater than zero, was "
        << max_sample_attempts;
    throw std::invalid_argument(oss.str());
  }
}

template <typename VecT>
struct ActiveSample {
  VecT position;
  std::size_t active_index;
  std::uint32_t sample_index;
};

/*!
Returns a pseudo-randomly selected active sample from the active indices.
*/
template <typename VecT>
ActiveSample<VecT> RandActiveSample(
    const std::vector<std::uint32_t>& active_indices,
    const std::vector<VecT>& samples, std::uint32_t* const seed) {
  auto active_sample = ActiveSample<VecT>{};
  active_sample.active_index = rand::IndexRand(active_indices.size(), seed);
  active_sample.sample_index = active_indices[active_sample.active_index];
  active_sample.position = samples[active_sample.sample_index];
  return active_sample;
}

/*!
Returns a pseudo-random coordinate that is guaranteed be at a
distance [@a radius, 2 * @a radius] from @a center.
*/
template <typename VecTraitsT, typename VecT, typename FloatT>
VecT RandAnnulusSample(const VecT& center, const FloatT radius,
                       std::uint32_t* const seed) {
  typedef typename VecTraitsT::ValueType VecValueType;

  assert(seed != nullptr && "seed is null");

  // Initialize bounds.
  auto rand_min = std::array<FloatT, VecTraitsT::kSize>{};
  auto rand_max = std::array<FloatT, VecTraitsT::kSize>{};
  std::fill(std::begin(rand_min), std::end(rand_min), FloatT{-2});
  std::fill(std::begin(rand_max), std::end(rand_max), FloatT{2});

  auto p = VecT{};
  for (;;) {
    // Generate a random component in the range [-2, 2] for each dimension.
    const auto offset = rand::ArrayRangeRand(rand_min, rand_max, seed);

    // The randomized offset is not guaranteed to be within the radial
    // distance that we need to guarantee. If we found an offset with
    // magnitude in the range (1, 2] we are done, otherwise generate a new
    // offset. Continue until a valid offset is found.
    const auto r2 = util::SquaredMagnitude(offset);
    if (FloatT{1} < r2 && r2 <= FloatT{4}) {
      // Found a valid offset.
      // Add the offset scaled by radius to the center coordinate to
      // produce the final sample.
      for (auto i = std::size_t{0}; i < VecTraitsT::kSize; ++i) {
        const auto pi = static_cast<FloatT>(VecTraitsT::Get(center, i)) +
                        radius * offset[i];
        VecTraitsT::Set(&p, i, static_cast<VecValueType>(pi));
      }
      break;
    }
  }
  return p;
}

/*!
Add a new sample and update data structures.
*/
template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
void AddSample(const VecT& sample, std::vector<VecT>* const samples,
               std::vector<std::uint32_t>* const active_indices,
               grid::Grid<FloatT, N>* const grid) {
  const auto sample_index = samples->size();  // New sample index.
  samples->push_back(sample);
  active_indices->push_back(static_cast<std::uint32_t>(sample_index));
  const auto grid_index = grid->template IndexFromSample<VecTraitsT>(sample);
  grid->Cell(grid_index) =
      static_cast<std::int32_t>(sample_index);  // No range check!
}

template <typename IndexT>
struct GridIndexRange {
  IndexT min_index;
  IndexT max_index;
};

/*!
Returns the grid neighborhood of the sample using the radius of the grid.
*/
template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
GridIndexRange<typename grid::Grid<FloatT, N>::IndexType> GridNeighborhood(
    const VecT& sample, const grid::Grid<FloatT, N>& grid) {
  typedef typename std::decay<decltype(grid)>::type::IndexType GridIndexType;
  typedef typename GridIndexType::value_type GridIndexValueType;

  constexpr auto kDims = std::tuple_size<GridIndexType>::value;
  static_assert(VecTraitsT::kSize == kDims, "dimensionality mismatch");

  auto min_index = GridIndexType{};
  auto max_index = GridIndexType{};
  const auto grid_size = grid.size();
  const auto radius = grid.sample_radius();
  for (auto i = std::size_t{0}; i < kDims; ++i) {
    const auto xi_min = GridIndexValueType{0};
    const auto xi_max = static_cast<GridIndexValueType>(grid_size[i] - 1);
    const auto xi = static_cast<FloatT>(VecTraitsT::Get(sample, i));
    const auto xi_sub = grid.AxisIndex(i, xi - radius);
    const auto xi_add = grid.AxisIndex(i, xi + radius);
    min_index[i] = util::clamp(xi_min, xi_max, xi_sub);
    max_index[i] = util::clamp(xi_min, xi_max, xi_add);
  }
  return GridIndexRange<GridIndexType>{min_index, max_index};
}

/*!
Returns true if there exists another sample within the radius used to
construct the grid, otherwise false.
*/
template <typename VecTraitsT, typename VecT, typename FloatT, std::size_t N>
bool ExistingSampleWithinRadius(
    const VecT& sample, const std::uint32_t active_sample_index,
    const std::vector<VecT>& samples, const grid::Grid<FloatT, N>& grid,
    const typename grid::Grid<FloatT, N>::IndexType& min_index,
    const typename grid::Grid<FloatT, N>::IndexType& max_index) {
  auto index = min_index;
  do {
    const auto cell_index = grid.Cell(index);
    if (cell_index >= 0 &&
        static_cast<std::uint32_t>(cell_index) != active_sample_index) {
      const auto cell_sample = samples[static_cast<std::uint32_t>(cell_index)];
      const auto d = static_cast<FloatT>(
          util::SquaredDistance<VecTraitsT>(sample, cell_sample));
      if (d < util::squared(grid.sample_radius())) {
        return true;
      }
    }
  } while (util::Iterate(min_index, max_index, &index));

  return false;
}

}  // namespace detail

/*!
Generic template for vector traits. Users may specialize this template
for their own classes.

Specializations must have the following static interface.

struct MyVecTraits<MyVec>
{
  typedef ... ValueType
  static constexpr std::size_t kSize = ...

  static ValueType Get(const MyVec&, std::size_t)
  static void Get(MyVec* const, std::size_t, ValueType)
}

See specialization for std::array below as an example.
*/
template <typename VecT>
struct VecTraits;

/*!
Specialization of vector traits for std::array.
*/
template <typename FloatT, std::size_t N>
struct VecTraits<std::array<FloatT, N>> {
  typedef typename std::array<FloatT, N>::value_type ValueType;

  static constexpr auto kSize = std::tuple_size<std::array<FloatT, N>>::value;

  static ValueType Get(const std::array<FloatT, N>& vec, const std::size_t i) {
    assert(i < kSize && "index out of bounds");
    return vec[i];
  }

  static void Set(std::array<FloatT, N>* const vec, const std::size_t i,
                  const ValueType value) {
    assert(i < kSize && "index out of bounds");
    (*vec)[i] = value;
  }
};

/*!
Returns a list of samples that are guaranteed to:
  * No two samples are closer to each other than radius.
  * No sample is outside the region [x_min, x_max].
*/
template <typename FloatT, std::size_t N, typename VecT = std::array<FloatT, N>,
          typename VecTraitsT = VecTraits<VecT>>
std::vector<VecT> PoissonDiskSampling(
    const FloatT radius, const std::array<FloatT, N>& x_min,
    const std::array<FloatT, N>& x_max,
    const std::uint32_t max_sample_attempts = 30,
    const std::uint32_t seed = 0) {
  using namespace detail;

  // Validate input.
  ThrowIfInvalidRadius(radius);
  ThrowIfInvalidBounds(x_min, x_max);
  ThrowIfInvalidMaxSampleAttempts(max_sample_attempts);

  // Acceleration grid.
  auto grid = grid::MakeGrid(radius, x_min, x_max);

  auto samples = std::vector<VecT>{};
  auto active_indices = std::vector<std::uint32_t>{};
  auto local_seed = seed;

  // Add first sample randomly within bounds.
  // No need to check (non-existing) neighbors.
  AddSample<VecTraitsT>(
      rand::VecRangeRand<VecT, VecTraitsT>(x_min, x_max, &local_seed), &samples,
      &active_indices, &grid);

  while (!active_indices.empty()) {
    // Randomly choose an active sample. A sample is considered active
    // until failed attempts have been made to generate a new sample within
    // its annulus.
    const auto active_sample =
        RandActiveSample(active_indices, samples, &local_seed);

    auto attempt_count = decltype(max_sample_attempts){0};
    while (attempt_count < max_sample_attempts) {
      // Randomly create a candidate sample inside the active sample's annulus.
      const auto cand_sample = RandAnnulusSample<VecTraitsT>(
          active_sample.position, grid.sample_radius(), &local_seed);

      // Check if candidate sample is within bounds.
      if (util::Inside<VecTraitsT>(cand_sample, x_min, x_max)) {
        // Check candidate sample proximity to nearby samples.
        const auto grid_neighbors =
            GridNeighborhood<VecTraitsT>(cand_sample, grid);
        const auto existing_sample = ExistingSampleWithinRadius<VecTraitsT>(
            cand_sample, active_sample.sample_index, samples, grid,
            grid_neighbors.min_index, grid_neighbors.max_index);
        if (!existing_sample) {
          // No existing samples where found to be too close to the
          // candidate sample, no further attempts necessary.
          AddSample<VecTraitsT>(cand_sample, &samples, &active_indices, &grid);
          break;
        }
        // Else: The candidate sample is too close to an existing sample,
        // move on to next attempt.
      }
      // Else: The candidate sample is out-of-bounds.

      ++attempt_count;
    }

    if (attempt_count == max_sample_attempts) {
      // No valid sample was found on the disk of the active sample,
      // remove it from the active list.
      util::EraseUnordered(&active_indices, active_sample.active_index);
    }
  }

  return samples;
}

}  // namespace poisson_disk_sampling
}  // namespace thinks

#endif  // THINKS_POISSON_DISK_SAMPLING_POISSON_DISK_SAMPLING_H_INCLUDED
