#pragma once
// -----------------------------------------------------------------------------
// Pure bid-utility computation for the decentralized allocator.
//
// This header has NO ROS dependency, on purpose. Two reasons:
//
//   1. It is the ONLY place the `distance` vs `confidence_energy` ablation is
//      allowed to exist. Keeping it in one small, dependency-free file makes
//      that claim auditable instead of aspirational. If you ever branch on
//      bid_mode in another translation unit, the comparison is no longer clean
//      and the experiment is void.
//
//   2. Without rclcpp it compiles with plain g++, so test_utility.cpp runs in
//      CI on a machine that has never had ROS 2 installed.
//
// Changes from the original inline Allocator::utility():
//   * bid_mode is parsed ONCE via parse_bid_mode() and stored as an enum.
//     The old version threw std::runtime_error from inside utility(), which is
//     reached from a subscription callback -- an exception escaping a callback
//     propagates out of spin() and terminates the node mid-run. You would lose
//     an overnight batch and find out from an empty CSV.
//   * The state-of-charge term is now explicit about the "no energy monitor
//     yet" case. The old expression energy_j_ / std::max(1.0, energy_capacity_j_)
//     evaluated to 1.0/1.0 with the defaults (1.0, -1.0) and that was invisible.
// -----------------------------------------------------------------------------

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
  ConfidenceEnergy,  // deliberately deviates from distance-optimal
};

/// Parse at construction time, never inside a callback.
inline BidMode parse_bid_mode(const std::string & s)
{
  if (s == "distance") {return BidMode::Distance;}
  if (s == "confidence_energy") {return BidMode::ConfidenceEnergy;}
  throw std::invalid_argument(
          "unknown bid_mode: '" + s + "' (expected 'distance' or 'confidence_energy')");
}

inline const char * to_string(BidMode m)
{
  return m == BidMode::Distance ? "distance" : "confidence_energy";
}

/// Sentinel for "this robot cannot serve this task". Must sort below any
/// feasible utility in either mode; field diagonals are ~30 m, so -1e9 is safe.
constexpr double kInfeasibleUtility = -1.0e9;

struct BidInputs
{
  double distance_m = 0.0;        ///< euclidean robot -> task
  double confidence = 0.0;        ///< fused detector confidence, [0, 1]
  double energy_cost_j = 0.0;     ///< predicted cost to drive there and treat
  double energy_j = 0.0;          ///< remaining energy
  double energy_capacity_j = 0.0; ///< pack capacity
  /// False until an energy monitor publishes RobotState. While false the SoC
  /// term is 1.0 by documented contract, so confidence_energy degenerates to
  /// confidence / cost. Any result produced with energy_known == false must
  /// say so in the README.
  bool energy_known = false;
  bool feasible = true;           ///< reserve-fraction gate, computed by caller
};

/// State of charge clamped to [0, 1].
inline double state_of_charge(const BidInputs & in)
{
  if (!in.energy_known || in.energy_capacity_j <= 0.0) {return 1.0;}
  return std::min(1.0, std::max(0.0, in.energy_j / in.energy_capacity_j));
}

/// The ablation. Nothing else in the repo may branch on BidMode.
///
/// conf_gamma defaults to 1.0. The original 1.5 had no justification; sweeping
/// an unjustified exponent costs runs and buys no argument. Leave it at 1.0
/// unless you can defend a different value in the README.
inline double utility(BidMode mode, const BidInputs & in, double conf_gamma = 1.0)
{
  if (!in.feasible) {return kInfeasibleUtility;}

  switch (mode) {
    case BidMode::Distance:
      // Reads ONLY distance. The unit test asserts this by construction: it is
      // what makes the baseline a real baseline rather than a variant.
      return -in.distance_m;

    case BidMode::ConfidenceEnergy: {
        const double c = std::pow(std::max(1e-3, in.confidence), conf_gamma);
        const double soc = state_of_charge(in);
        return (c * soc) / (in.energy_cost_j + 1.0);
      }
  }
  return kInfeasibleUtility;  // unreachable; silences -Wreturn-type
}

/// Deterministic winner comparison: higher utility wins, ties broken by LOWER
/// robot id. Every robot computes this independently over the bids it happened
/// to receive; determinism here is what keeps split-brain rare under loss, and
/// TaskAward.n_bids_seen is what makes the residual disagreement visible.
/// Templated on the id type: the node currently uses std::string ids
/// ("robot_0"), which order lexicographically -- so "robot_10" sorts before
/// "robot_2". That is ugly but DETERMINISTIC, which is all this needs to be.
/// If you ever switch to numeric ids, this keeps working unchanged.
template<typename Id>
inline bool bid_beats(double util_a, const Id & id_a, double util_b, const Id & id_b)
{
  if (util_a != util_b) {return util_a > util_b;}
  return id_a < id_b;
}

}  // namespace agri_swarm
