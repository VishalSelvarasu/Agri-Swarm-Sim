#include <cstdio>
#include <cstdlib>
#include <string>

#include "agri_swarm_allocation/utility.hpp"

using namespace agri_swarm;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond) \
  do { \
    ++g_checks; \
    if (!(cond)) { \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures; \
    } \
  } while (0)

static BidInputs base()
{
  BidInputs in;
  in.distance_m = 5.0;
  in.confidence = 0.6;
  in.energy_cost_j = 100.0;
  in.energy_j = 20000.0;
  in.energy_capacity_j = 40000.0;
  in.energy_known = true;
  in.feasible = true;
  return in;
}

// ---------------------------------------------------------------------------
// Mode parsing happens once, at construction, and rejects garbage loudly.
// ---------------------------------------------------------------------------
static void test_parse_bid_mode()
{
  std::printf("parse_bid_mode\n");
  CHECK(parse_bid_mode("distance") == BidMode::Distance);
  CHECK(parse_bid_mode("confidence_energy") == BidMode::ConfidenceEnergy);

  bool threw = false;
  try {
    parse_bid_mode("confidence-energy");
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try {
    parse_bid_mode("");
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

// ---------------------------------------------------------------------------
// THE ablation invariant. Distance mode must be a pure function of distance.
// If someone quietly folds confidence into the baseline to make the headline
// number nicer, this test is what catches it.
// ---------------------------------------------------------------------------
static void test_distance_mode_ignores_everything_else()
{
  std::printf("distance mode is a function of distance ALONE\n");
  BidInputs a = base();
  BidInputs b = base();
  b.confidence = 0.01;
  b.energy_cost_j = 9999.0;
  b.energy_j = 1.0;
  b.energy_capacity_j = 40000.0;

  CHECK(utility(BidMode::Distance, a) == utility(BidMode::Distance, b));

  // ...and strictly decreasing in distance, so nearest robot always wins.
  BidInputs near = base(); near.distance_m = 1.0;
  BidInputs far = base(); far.distance_m = 12.0;
  CHECK(utility(BidMode::Distance, near) > utility(BidMode::Distance, far));

  // conf_gamma must not touch the baseline either.
  CHECK(utility(BidMode::Distance, a, 1.0) == utility(BidMode::Distance, a, 3.7));
}

// ---------------------------------------------------------------------------
static void test_infeasible_dominates_in_both_modes()
{
  std::printf("infeasible sorts below every feasible bid\n");
  BidInputs bad = base();
  bad.feasible = false;

  CHECK(utility(BidMode::Distance, bad) == kInfeasibleUtility);
  CHECK(utility(BidMode::ConfidenceEnergy, bad) == kInfeasibleUtility);

  // Worst realistic feasible distance bid still beats infeasible.
  BidInputs worst = base();
  worst.distance_m = 1000.0;
  CHECK(utility(BidMode::Distance, worst) > kInfeasibleUtility);
}

// ---------------------------------------------------------------------------
static void test_confidence_energy_monotonicity()
{
  std::printf("confidence_energy monotonicity\n");

  BidInputs lo = base(); lo.confidence = 0.3;
  BidInputs hi = base(); hi.confidence = 0.9;
  CHECK(utility(BidMode::ConfidenceEnergy, hi) > utility(BidMode::ConfidenceEnergy, lo));

  BidInputs cheap = base(); cheap.energy_cost_j = 10.0;
  BidInputs dear = base(); dear.energy_cost_j = 500.0;
  CHECK(utility(BidMode::ConfidenceEnergy, cheap) > utility(BidMode::ConfidenceEnergy, dear));

  BidInputs full = base(); full.energy_j = 39000.0;
  BidInputs flat = base(); flat.energy_j = 4000.0;
  CHECK(utility(BidMode::ConfidenceEnergy, full) > utility(BidMode::ConfidenceEnergy, flat));

  // Zero confidence is floored, not divided-by-zero or negative.
  BidInputs zero = base(); zero.confidence = 0.0;
  CHECK(utility(BidMode::ConfidenceEnergy, zero) > 0.0);
}

// ---------------------------------------------------------------------------
// The degenerate state the repo is in RIGHT NOW: no energy monitor exists, so
// energy_known is false everywhere. This test pins the contract so the
// degeneracy is a documented fact rather than an accident of std::max.
// ---------------------------------------------------------------------------
static void test_soc_contract_without_energy_monitor()
{
  std::printf("state_of_charge contract\n");

  BidInputs unknown = base();
  unknown.energy_known = false;
  CHECK(state_of_charge(unknown) == 1.0);

  BidInputs bad_capacity = base();
  bad_capacity.energy_capacity_j = -1.0;  // the old default
  CHECK(state_of_charge(bad_capacity) == 1.0);

  BidInputs over = base();
  over.energy_j = 99999.0;
  CHECK(state_of_charge(over) == 1.0);       // clamped

  BidInputs negative = base();
  negative.energy_j = -5.0;
  CHECK(state_of_charge(negative) == 0.0);   // clamped

  BidInputs half = base();
  half.energy_j = 20000.0;
  half.energy_capacity_j = 40000.0;
  CHECK(state_of_charge(half) == 0.5);
}

// ---------------------------------------------------------------------------
// Winner selection must be a strict total order, or two robots that received
// the same bid set can still disagree -- which would make the reported
// split-brain rate a measure of your comparator, not of message loss.
// ---------------------------------------------------------------------------
static void test_tie_break_is_a_total_order()
{
  std::printf("bid_beats is deterministic and antisymmetric\n");

  CHECK(bid_beats(2.0, 7, 1.0, 0));          // higher utility wins
  CHECK(!bid_beats(1.0, 0, 2.0, 7));
  CHECK(bid_beats(1.0, 3, 1.0, 9));          // tie -> lower id
  CHECK(!bid_beats(1.0, 9, 1.0, 3));
  CHECK(!bid_beats(1.0, 4, 1.0, 4));         // irreflexive

  // Exhaustive antisymmetry + determinism over a small grid.
  const double utils[] = {-1.0e9, -5.0, 0.0, 0.25, 3.0};
  for (uint32_t ia = 0; ia < 4; ++ia) {
    for (uint32_t ib = 0; ib < 4; ++ib) {
      for (const double ua : utils) {
        for (const double ub : utils) {
          const bool ab = bid_beats(ua, ia, ub, ib);
          const bool ba = bid_beats(ub, ib, ua, ia);
          if (ia == ib && ua == ub) {
            CHECK(!ab && !ba);
          } else {
            CHECK(ab != ba);
          }
          CHECK(bid_beats(ua, ia, ub, ib) == ab);  // pure
        }
      }
    }
  }
}

int main()
{
  test_parse_bid_mode();
  test_distance_mode_ignores_everything_else();
  test_infeasible_dominates_in_both_modes();
  test_confidence_energy_monotonicity();
  test_soc_contract_without_energy_monitor();
  test_tie_break_is_a_total_order();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}