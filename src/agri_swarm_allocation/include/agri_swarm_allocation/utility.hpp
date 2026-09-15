#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace agri_swarm
{

enum class BidMode
{
  Distance,          // classic nearest-robot; distance-OPTIMAL by construction
  // Ranks bidders by state of charge over estimated energy cost.
  //
  // Named ConfidenceEnergy until the ablation was analysed. Task
  // confidence multiplies the whole expression, and within one auction
  // every bidder sees the same task, so the factor is identical across
  // bids and cancels from the argmax. It is retained because it scales
  // utilities comparably across tasks, but it cannot change which robot
  // wins one. Confidence gates whether a task exists at all, via
  // treat_confidence_threshold; it does not decide who services it.
  // See test_energy_aware_is_rank_invariant_in_confidence.
  EnergyAware,
};

/// Parse at construction time, never inside a callback.
inline BidMode parse_bid_mode(const std::string & s)
{
  if (s == "distance") {return BidMode::Distance;}
  if (s == "energy_aware") {return BidMode::EnergyAware;}
  // Accepted so that run directories and results.csv rows written before
  // the rename still parse against current code.
  if (s == "confidence_energy") {return BidMode::EnergyAware;}
  throw std::invalid_argument(
          "unknown bid_mode: '" + s + "' (expected 'distance' or 'energy_aware')");
}

inline const char * to_string(BidMode m)
{
  return m == BidMode::Distance ? "distance" : "energy_aware";
}


constexpr double kInfeasibleUtility = -1.0e9;

struct BidInputs
{
  double distance_m = 0.0;        ///< euclidean robot -> task
  double confidence = 0.0;        ///< fused detector confidence, [0, 1]
  double energy_cost_j = 0.0;     ///< predicted cost to drive there and treat
  double energy_j = 0.0;          ///< remaining energy
  double energy_capacity_j = 0.0; ///< pack capacity
  bool energy_known = false;
  bool feasible = true;           ///< reserve-fraction gate, computed by caller
};

/// State of charge clamped to [0, 1].
inline double state_of_charge(const BidInputs & in)
{
  if (!in.energy_known || in.energy_capacity_j <= 0.0) {return 1.0;}
  return std::min(1.0, std::max(0.0, in.energy_j / in.energy_capacity_j));
}


inline double utility(BidMode mode, const BidInputs & in, double conf_gamma = 1.0)
{
  if (!in.feasible) {return kInfeasibleUtility;}

  switch (mode) {
    case BidMode::Distance:

      return -in.distance_m;

    case BidMode::EnergyAware: {
        const double c = std::pow(std::max(1e-3, in.confidence), conf_gamma);
        const double soc = state_of_charge(in);
        return (c * soc) / (in.energy_cost_j + 1.0);
      }
  }
  return kInfeasibleUtility;  // unreachable; silences -Wreturn-type
}

template<typename Id>
inline bool bid_beats(double util_a, const Id & id_a, double util_b, const Id & id_b)
{
  if (util_a != util_b) {return util_a > util_b;}
  return id_a < id_b;
}

inline uint32_t task_id_for(double x, double y, double cell)
{
  const int64_t ix = static_cast<int64_t>(std::floor(x / cell));
  const int64_t iy = static_cast<int64_t>(std::floor(y / cell));
  uint64_t h = 1469598103934665603ULL;                 // FNV-1a
  for (int64_t v : {ix, iy}) {
    for (int b = 0; b < 8; ++b) {
      h ^= static_cast<uint8_t>((v >> (b * 8)) & 0xFF);
      h *= 1099511628211ULL;
    }
  }
  return static_cast<uint32_t>(h ^ (h >> 32));
}

}  