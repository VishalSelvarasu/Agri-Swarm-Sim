#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>
 
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
 
#include <agri_swarm_allocation/utility.hpp>
 
#include <agri_swarm_msgs/msg/bid.hpp>
#include <agri_swarm_msgs/msg/robot_state.hpp>
#include <agri_swarm_msgs/msg/task_announcement.hpp>
#include <agri_swarm_msgs/msg/task_award.hpp>
#include <agri_swarm_msgs/msg/weed_detection.hpp>
 
using namespace std::chrono_literals;
namespace M = agri_swarm_msgs::msg;
 
namespace
{
 
/// Quantises a position onto a grid so that independent sightings of one patch
/// produce the same identifier. The cell size must exceed the detector's
/// position noise, or a single patch fragments into several tasks.
uint32_t task_id_for(double x, double y, double cell)
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
 
enum class Phase { Announced, Awarded, Done };
 
/// Cause of a re-announcement. The three cases are distinct failures and are
/// reported separately: an empty bid set, a bid set containing no feasible
/// entry, and a winner that stopped sending heartbeats.
enum class Reason { NoBids, AllInfeasible, SilentWinner };
 
const char * reason_str(Reason r)
{
  switch (r) {
    case Reason::NoBids:        return "no bids received";
    case Reason::AllInfeasible: return "no feasible bidder";
    case Reason::SilentWinner:  return "winner silent";
  }
  return "unknown";
}
 
struct Task
{
  bool init{false};
  uint32_t id{};
  double x{}, y{}, radius{};
  double confidence{};                  ///< Fused over sightings.
  uint32_t round{};
  Phase phase{Phase::Announced};
  rclcpp::Time deadline;
  std::string winner;
  std::vector<M::Bid> bids;             ///< Bids seen for the current round.
};
 
}  // namespace
 
 
class Allocator : public rclcpp::Node
{
public:
  Allocator()
  : rclcpp::Node("allocator")
  {
    robot_id_ = declare_parameter<std::string>("robot_id", "robot_0");
    // Parsed once at construction so that an invalid value cannot propagate out
    // of a subscription callback.
    bid_mode_ = agri_swarm::parse_bid_mode(
      declare_parameter<std::string>("bid_mode", "confidence_energy"));
 
    cell_         = declare_parameter<double>("task_cell_size", 0.30);
    bid_window_   = declare_parameter<double>("bid_window_s", 0.6);
    award_grace_  = declare_parameter<double>("award_grace_s", 1.2);
    max_rounds_   = declare_parameter<int>("max_rounds", 4);
    energy_per_m_ = declare_parameter<double>("energy_per_m_j", 12.0);
    treat_cost_j_ = declare_parameter<double>("treat_cost_j", 30.0);
    reserve_frac_ = declare_parameter<double>("reserve_fraction", 0.15);
    conf_gamma_   = declare_parameter<double>("confidence_gamma", 1.0);
    // Detections below this never become tasks. The detector emits false
    // positives at a fixed rate regardless of what is on the ground, so
    // without a floor the auction announces noise faster than the swarm can
    // service it and the mission does not terminate. The offline threshold
    // sweep in score_run.py is valid only from this value upward.
    treat_conf_threshold_ =
      declare_parameter<double>("treat_confidence_threshold", 0.2);
    origin_x_     = declare_parameter<double>("origin_x", 0.0);
    origin_y_     = declare_parameter<double>("origin_y", 0.0);
 
    // Configured rather than inferred from the first heartbeat, so that results
    // do not depend on discovery order. The energy monitor must be launched
    // with the same value. The state-of-charge term and the reserve gate are
    // only informative if capacity is of the same order as mission consumption.
    energy_capacity_j_ = declare_parameter<double>("energy_capacity_j", 40000.0);
 
    if (robot_id_ != "robot_0" && origin_x_ == 0.0 && origin_y_ == 0.0) {
      RCLCPP_WARN(get_logger(),
                  "%s: origin_x and origin_y are both zero, so bid distances "
                  "will be computed in the wrong frame unless the launch file "
                  "supplies the spawn pose",
                  robot_id_.c_str());
    }
 
    // Deterministic per robot, to break announcement symmetry reproducibly.
    rng_.seed(std::hash<std::string>{}(robot_id_));
 
    auto qos = rclcpp::QoS(50);
    pub_ann_   = create_publisher<M::TaskAnnouncement>("/task_announcements", qos);
    pub_bid_   = create_publisher<M::Bid>("/bids", qos);
    pub_award_ = create_publisher<M::TaskAward>("/task_awards", qos);
 
    sub_det_ = create_subscription<M::WeedDetection>(
      "/weed_detections", qos, [this](M::WeedDetection::SharedPtr m) { onDetection(*m); });
    sub_ann_ = create_subscription<M::TaskAnnouncement>(
      "/task_announcements", qos, [this](M::TaskAnnouncement::SharedPtr m) { onAnnouncement(*m); });
    sub_bid_ = create_subscription<M::Bid>(
      "/bids", qos, [this](M::Bid::SharedPtr m) { onBid(*m); });
    sub_award_ = create_subscription<M::TaskAward>(
      "/task_awards", qos, [this](M::TaskAward::SharedPtr m) { onAward(*m); });
    sub_state_ = create_subscription<M::RobotState>(
      "/robot_states", qos, [this](M::RobotState::SharedPtr m) { onState(*m); });
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "odom", 10, [this](nav_msgs::msg::Odometry::SharedPtr m) { onOdom(*m); });
 
    timer_ = create_wall_timer(50ms, [this] { resolve(); });
  }
 
private:
  // --------------------------------------------------------------------- pose
 
  void onOdom(const nav_msgs::msg::Odometry & m)
  {
    x_ = origin_x_ + m.pose.pose.position.x;
    y_ = origin_y_ + m.pose.pose.position.y;
    have_pose_ = true;
 
    if (!logged_first_pose_) {
      logged_first_pose_ = true;
      RCLCPP_INFO(get_logger(),
                  "first pose: world (%.3f, %.3f) = odom (%.3f, %.3f) + origin (%.3f, %.3f)",
                  x_, y_,
                  m.pose.pose.position.x, m.pose.pose.position.y,
                  origin_x_, origin_y_);
    }
  }
 
  // ---------------------------------------------------------------- detection
 
  void onDetection(const M::WeedDetection & d)
  {
    if (d.confidence < treat_conf_threshold_) {
      return;
    }
 
    const uint32_t id = task_id_for(d.position.x, d.position.y, cell_);
    auto it = tasks_.find(id);
 
    if (it != tasks_.end()) {
      // Fusion rule: maximum confidence over sightings.
      it->second.confidence = std::max<double>(it->second.confidence, d.confidence);
      return;
    }
    if (done_.count(id)) {
      return;
    }
 
    Task t;
    t.init = true;
    t.id = id;
    t.x = d.position.x;
    t.y = d.position.y;
    t.radius = d.radius;
    t.confidence = d.confidence;
    t.round = 0;
    t.deadline = now() + rclcpp::Duration::from_seconds(bid_window_);
    tasks_[id] = t;
 
    // Announce after a short random delay. If a peer announces the same task
    // first, onAnnouncement cancels this one.
    std::uniform_real_distribution<double> jitter(0.0, 0.15);
    pending_announce_[id] = now() + rclcpp::Duration::from_seconds(jitter(rng_));
  }
 
  void onAnnouncement(const M::TaskAnnouncement & a)
  {
    // A node receives its own publications. Skipping them prevents the
    // announcer bidding twice and inflating n_bids_seen.
    if (a.announcer_id == robot_id_) {
      return;
    }
    // Applied here as well as in onDetection: a peer running a different
    // threshold must not be able to pull this robot below its own floor.
    if (a.confidence < treat_conf_threshold_) {
      return;
    }
    if (done_.count(a.task_id)) {
      return;
    }
    pending_announce_.erase(a.task_id);
 
    Task & t = tasks_[a.task_id];
    if (!t.init || t.round < a.round) {
      t.init = true;
      t.id = a.task_id;
      t.x = a.position.x;
      t.y = a.position.y;
      t.radius = a.radius;
      t.round = a.round;
      t.phase = Phase::Announced;
      t.bids.clear();
      t.winner.clear();
    }
    t.confidence = std::max<double>(t.confidence, a.confidence);
    // The clock type must match the node clock, or the comparison in resolve()
    // throws under use_sim_time.
    t.deadline = rclcpp::Time(a.bid_deadline, get_clock()->get_clock_type());
 
    submitBid(t);
  }
 
  // --------------------------------------------------------------------- bids
 
  void submitBid(const Task & t)
  {
    if (!have_pose_ || status_ == M::RobotState::STATUS_FAILED) {
      return;
    }
 
    const double dist = std::hypot(t.x - x_, t.y - y_);
    const double e_cost = dist * energy_per_m_ + treat_cost_j_;
    // A fraction of the pack is reserved for the return leg. While the pack
    // state is unknown, feasibility is reported as true rather than being
    // decided against an unset capacity.
    const bool feasible =
      !energy_known_ ||
      (energy_j_ - e_cost) > reserve_frac_ * energy_capacity_j_;
 
    M::Bid b;
    b.header.stamp = now();
    b.task_id = t.id;
    b.round = t.round;
    b.robot_id = robot_id_;
    b.travel_cost_m = static_cast<float>(dist);
    b.energy_cost_j = static_cast<float>(e_cost);
    b.remaining_energy_j = static_cast<float>(energy_j_);
    b.feasible = feasible;
    b.utility = static_cast<float>(utility(t, dist, e_cost, feasible));
    pub_bid_->publish(b);
  }
 
  /// Marshals node state into the ROS-free utility function in utility.hpp,
  /// which holds the distance / confidence_energy ablation.
  double utility(const Task & t, double dist, double e_cost, bool feasible) const
  {
    agri_swarm::BidInputs in;
    in.distance_m        = dist;
    in.confidence        = t.confidence;
    in.energy_cost_j     = e_cost;
    in.energy_j          = energy_j_;
    in.energy_capacity_j = energy_capacity_j_;
    in.energy_known      = energy_known_;
    in.feasible          = feasible;
    return agri_swarm::utility(bid_mode_, in, conf_gamma_);
  }
 
  void onBid(const M::Bid & b)
  {
    auto it = tasks_.find(b.task_id);
    if (it == tasks_.end() || it->second.round != b.round) {
      return;
    }
    if (it->second.phase != Phase::Announced) {
      return;
    }
    it->second.bids.push_back(b);
  }
 
  // ------------------------------------------------------------------ resolve
 
  void resolve()
  {
    const rclcpp::Time t_now = now();
 
    for (auto it = pending_announce_.begin(); it != pending_announce_.end(); ) {
      if (t_now >= it->second) {
        announce(tasks_[it->first]);
        it = pending_announce_.erase(it);
      } else {
        ++it;
      }
    }
 
    for (auto & [id, t] : tasks_) {
      if (t.phase == Phase::Announced && t_now >= t.deadline) {
        decide(t, t_now);
      } else if (t.phase == Phase::Awarded &&
                 t.winner != robot_id_ &&
                 isSilent(t.winner, t_now))
      {
        reannounce(t, t_now, Reason::SilentWinner);
      }
    }
  }
 
  void announce(Task & t)
  {
    M::TaskAnnouncement a;
    a.header.stamp = now();
    a.task_id = t.id;
    a.announcer_id = robot_id_;
    a.position.x = t.x;
    a.position.y = t.y;
    a.radius = static_cast<float>(t.radius);
    a.confidence = static_cast<float>(t.confidence);
    a.round = t.round;
    a.bid_deadline = t.deadline;
    pub_ann_->publish(a);
 
    submitBid(t);
  }
 
  void decide(Task & t, const rclcpp::Time & t_now)
  {
    if (t.bids.empty()) {
      reannounce(t, t_now, Reason::NoBids);
      return;
    }
 
    // Deterministic given the same bid set, with robot_id as tie-break.
    auto best = std::max_element(
      t.bids.begin(), t.bids.end(), [](const M::Bid & a, const M::Bid & b) {
        // max_element expects "a < b"; bid_beats expresses "a wins over b", so
        // the arguments are swapped.
        return agri_swarm::bid_beats(
          static_cast<double>(b.utility), b.robot_id,
          static_cast<double>(a.utility), a.robot_id);
      });
 
    if (static_cast<double>(best->utility) <= agri_swarm::kInfeasibleUtility / 10.0) {
      reannounce(t, t_now, Reason::AllInfeasible);
      return;
    }
 
    t.winner = best->robot_id;
    t.phase = Phase::Awarded;
 
    if (t.winner == robot_id_) {
      M::TaskAward aw;
      aw.header.stamp = t_now;
      aw.task_id = t.id;
      aw.round = t.round;
      aw.winner_id = robot_id_;
      aw.winning_utility = best->utility;
      aw.n_bids_seen = static_cast<uint8_t>(std::min<size_t>(255, t.bids.size()));
      pub_award_->publish(aw);
      committed_.push_back(t.id);   // TODO: hand to the path executor.
    }
  }
 
  void onAward(const M::TaskAward & aw)
  {
    auto it = tasks_.find(aw.task_id);
    if (it == tasks_.end()) {
      return;
    }
    Task & t = it->second;
    if (aw.round < t.round) {
      return;
    }
 
    // Two robots reached different winners from different bid sets. Both bid
    // counts are logged; their difference bounds the message loss.
    if (t.phase == Phase::Awarded && !t.winner.empty() && t.winner != aw.winner_id) {
      const std::string & keep = (aw.winner_id < t.winner) ? aw.winner_id : t.winner;
      RCLCPP_WARN(get_logger(),
                  "split-brain task=%u round=%u: local winner %s (%zu bids seen) "
                  "vs announced %s (%u bids seen), conceding to %s",
                  aw.task_id, aw.round,
                  t.winner.c_str(), t.bids.size(),
                  aw.winner_id.c_str(), aw.n_bids_seen,
                  keep.c_str());
      split_brain_count_++;
 
      // Lowest robot_id wins. Arbitrary, but identical on every robot.
      if (aw.winner_id < t.winner) {
        t.winner = aw.winner_id;
        committed_.erase(std::remove(committed_.begin(), committed_.end(), t.id),
                         committed_.end());
      }
      return;
    }
 
    t.winner = aw.winner_id;
    t.phase = Phase::Awarded;
  }
 
  void reannounce(Task & t, const rclcpp::Time & t_now, Reason why)
  {
    if (t.round + 1 >= static_cast<uint32_t>(max_rounds_)) {
      // task_id is a position hash, so the coordinates are logged to allow
      // cross-referencing against the ground-truth field.
      RCLCPP_WARN(get_logger(),
                  "task=%u abandoned at (%.2f, %.2f) conf=%.2f after %d rounds, "
                  "last cause: %s",
                  t.id, t.x, t.y, t.confidence, max_rounds_, reason_str(why));
      done_.insert(t.id);
      t.phase = Phase::Done;
      return;
    }
 
    RCLCPP_DEBUG(get_logger(), "task=%u re-announce round %u to %u, cause: %s",
                 t.id, t.round, t.round + 1u, reason_str(why));
 
    t.round++;
    t.bids.clear();
    t.winner.clear();
    t.phase = Phase::Announced;
    t.deadline = t_now + rclcpp::Duration::from_seconds(bid_window_);
    announce(t);
  }
 
  // -------------------------------------------------------------------- peers
 
  void onState(const M::RobotState & s)
  {
    last_seen_[s.robot_id] = now();
    if (s.robot_id != robot_id_) {
      return;
    }
 
    if (!energy_known_ && s.remaining_energy_j > energy_capacity_j_ * 1.01) {
      RCLCPP_WARN(get_logger(),
                  "energy monitor reports %.1f J against a configured capacity "
                  "of %.1f J; both must be set from the same value",
                  static_cast<double>(s.remaining_energy_j), energy_capacity_j_);
    }
 
    energy_j_ = s.remaining_energy_j;
    status_ = s.status;
    energy_known_ = true;
  }
 
  bool isSilent(const std::string & id, const rclcpp::Time & t_now) const
  {
    auto it = last_seen_.find(id);
    if (it == last_seen_.end()) {
      return true;
    }
    return (t_now - it->second).seconds() > award_grace_;
  }
 
  // -------------------------------------------------------------------- state
 
  std::string robot_id_;
  agri_swarm::BidMode bid_mode_{agri_swarm::BidMode::ConfidenceEnergy};
  double cell_{}, bid_window_{}, award_grace_{};
  int max_rounds_{};
  double energy_per_m_{}, treat_cost_j_{}, reserve_frac_{}, conf_gamma_{};
  double treat_conf_threshold_{};
 
  double x_{}, y_{};                 ///< World frame.
  double origin_x_{}, origin_y_{};   ///< Spawn pose, added to wheel odometry.
  bool have_pose_{false};
  bool logged_first_pose_{false};
 
  double energy_j_{1.0};             ///< Valid only while energy_known_ is true.
  double energy_capacity_j_{};       ///< Configured pack capacity.
  /// True once the energy monitor has reported. While false the state-of-charge
  /// term is 1.0 and the feasibility gate is inactive, by contract.
  bool energy_known_{false};
 
  uint8_t status_{M::RobotState::STATUS_IDLE};
  uint64_t split_brain_count_{0};    ///< TODO: not yet exported for scoring.
 
  std::map<uint32_t, Task> tasks_;
  std::map<uint32_t, rclcpp::Time> pending_announce_;
  std::map<std::string, rclcpp::Time> last_seen_;
  std::vector<uint32_t> committed_;
  std::set<uint32_t> done_;
  std::mt19937 rng_;
 
  rclcpp::Publisher<M::TaskAnnouncement>::SharedPtr pub_ann_;
  rclcpp::Publisher<M::Bid>::SharedPtr pub_bid_;
  rclcpp::Publisher<M::TaskAward>::SharedPtr pub_award_;
  rclcpp::Subscription<M::WeedDetection>::SharedPtr sub_det_;
  rclcpp::Subscription<M::TaskAnnouncement>::SharedPtr sub_ann_;
  rclcpp::Subscription<M::Bid>::SharedPtr sub_bid_;
  rclcpp::Subscription<M::TaskAward>::SharedPtr sub_award_;
  rclcpp::Subscription<M::RobotState>::SharedPtr sub_state_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::TimerBase::SharedPtr timer_;
};
 
 
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Allocator>());
  rclcpp::shutdown();
  return 0;
}
