// Decentralized task allocator. One instance runs on every robot; there is no
// broker and no leader.
//
// The charter described this as "auction/consensus where bids = f(confidence,
// energy, distance)". That is one line, and it hides every part that is
// actually hard. This file pins down the parts that will otherwise surface as
// race conditions in week 5:
//
//   1. Task identity      -- two robots seeing the same weed must produce the
//                            SAME task_id, or you allocate it twice. Solved by
//                            hashing the quantized position, not by counters.
//   2. Announcement storms -- N robots seeing one patch must not emit N
//                            announcements. Solved by jittered announce +
//                            suppression on first-heard.
//   3. Winner selection    -- with no broker, every robot computes the winner
//                            independently from the bids IT received. Under
//                            message loss those sets differ. Deterministic
//                            tie-break by (utility, robot_id) makes disagreement
//                            rare; TaskAward with n_bids_seen makes it VISIBLE.
//   4. Silent winners      -- a robot that fails after winning must not strand
//                            its task. Solved by round-based re-announcement.
//
// Point 3 is the honest, testable core of this project. Run it under injected
// packet loss and report the split-brain rate; that experiment is worth more
// than the entire scalability study.
//
// Written in C++ on purpose: most robotics job ads in the DE market list C++ as
// a hard requirement, and this is the one component worth showing in it.

#include <algorithm>
#include <chrono>
#include <cmath>
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

namespace {

// Quantize to a grid so independent sightings of one patch collide.
// Cell size must exceed detector position_sigma by a healthy margin or the
// same weed fragments into several tasks.
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

struct Task
{
  bool init{false};
  uint32_t id{};
  double x{}, y{}, radius{};
  double confidence{};                  // fused over sightings
  uint32_t round{};
  Phase phase{Phase::Announced};
  rclcpp::Time deadline;
  std::string winner;
  std::vector<M::Bid> bids;             // bids seen for the CURRENT round
};

}  // namespace


class Allocator : public rclcpp::Node
{
public:
  Allocator() : rclcpp::Node("allocator")
  {
    robot_id_      = declare_parameter<std::string>("robot_id", "robot_0");
    // Parsed here and never again. The previous version compared the raw
    // string inside utility(), which is reached from a subscription callback;
    // an unknown mode threw out of spin() and killed the node mid-run.
    bid_mode_ = agri_swarm::parse_bid_mode(
      declare_parameter<std::string>("bid_mode", "confidence_energy"));
    cell_          = declare_parameter<double>("task_cell_size", 0.30);
    bid_window_    = declare_parameter<double>("bid_window_s", 0.6);
    award_grace_   = declare_parameter<double>("award_grace_s", 1.2);
    max_rounds_    = declare_parameter<int>("max_rounds", 4);
    energy_per_m_  = declare_parameter<double>("energy_per_m_j", 12.0);
    treat_cost_j_  = declare_parameter<double>("treat_cost_j", 30.0);
    reserve_frac_  = declare_parameter<double>("reserve_fraction", 0.15);
    conf_gamma_    = declare_parameter<double>("confidence_gamma", 1.0);

    // Jitter breaks announcement symmetry deterministically per robot.
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
      "odom", 10, [this](nav_msgs::msg::Odometry::SharedPtr m) {
        x_ = m->pose.pose.position.x;
        y_ = m->pose.pose.position.y;
        have_pose_ = true;
      });

    timer_ = create_wall_timer(50ms, [this] { resolve(); });
  }

private:
  // ---------------------------------------------------------------- detection

  void onDetection(const M::WeedDetection & d)
  {
    const uint32_t id = task_id_for(d.position.x, d.position.y, cell_);
    auto it = tasks_.find(id);

    if (it != tasks_.end()) {
      // Fuse. Naive max; a log-odds fuse is the obvious upgrade and is a
      // legitimate thing to ablate.
      it->second.confidence = std::max<double>(it->second.confidence, d.confidence);
      return;
    }
    if (done_.count(id)) return;

    Task t;
    t.init = true;
    t.id = id;
    t.x = d.position.x; t.y = d.position.y; t.radius = d.radius;
    t.confidence = d.confidence;
    t.round = 0;
    t.deadline = now() + rclcpp::Duration::from_seconds(bid_window_);
    tasks_[id] = t;

    // Jittered announce: if someone else announces this task first, our
    // pending announce is suppressed by onAnnouncement setting `announced`.
    std::uniform_real_distribution<double> j(0.0, 0.15);
    const double delay = j(rng_);
    pending_announce_[id] = now() + rclcpp::Duration::from_seconds(delay);
  }

  void onAnnouncement(const M::TaskAnnouncement & a)
  {
    if (done_.count(a.task_id)) return;
    pending_announce_.erase(a.task_id);          // someone beat us to it

    Task & t = tasks_[a.task_id];
    if (!t.init || t.round < a.round) {          // new task, or newer round
      t.init = true;
      t.id = a.task_id;
      t.x = a.position.x; t.y = a.position.y; t.radius = a.radius;
      t.round = a.round;
      t.phase = Phase::Announced;
      t.bids.clear();
      t.winner.clear();
    }
    t.confidence = std::max<double>(t.confidence, a.confidence);
    // Clock type MUST match the node clock or the comparison in resolve()
    // throws under use_sim_time. This is a real bug, not a nicety.
    t.deadline = rclcpp::Time(a.bid_deadline, get_clock()->get_clock_type());

    submitBid(t);
  }

  // -------------------------------------------------------------------- bids

  void submitBid(const Task & t)
  {
    if (!have_pose_ || status_ == M::RobotState::STATUS_FAILED) return;

    const double dist = std::hypot(t.x - x_, t.y - y_);
    const double e_cost = dist * energy_per_m_ + treat_cost_j_;
    // Round trip: must still be able to reach the headland afterwards.
    // Until an energy monitor publishes RobotState there is nothing to gate on,
    // so everything is feasible AND SAYS SO, rather than passing by accident
    // through a comparison against a -1.0 capacity.
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

  // The ablation itself now lives in utility.hpp, with no ROS dependency, so
  // it is unit-tested in CI on a machine that has never had ROS 2 installed.
  // This function only marshals node state into that pure call.
  double utility(const Task & t, double dist, double e_cost, bool feasible) const
  {
    agri_swarm::BidInputs in;
    in.distance_m       = dist;
    in.confidence       = t.confidence;
    in.energy_cost_j    = e_cost;
    in.energy_j         = energy_j_;
    in.energy_capacity_j = energy_capacity_j_;
    in.energy_known     = energy_known_;
    in.feasible         = feasible;
    return agri_swarm::utility(bid_mode_, in, conf_gamma_);
  }

  void onBid(const M::Bid & b)
  {
    auto it = tasks_.find(b.task_id);
    if (it == tasks_.end() || it->second.round != b.round) return;
    if (it->second.phase != Phase::Announced) return;
    it->second.bids.push_back(b);
  }

  // ----------------------------------------------------------------- resolve

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
                 isSilent(t.winner, t_now)) {
        reannounce(t, t_now);
      }
    }
  }

  void announce(Task & t)
  {
    M::TaskAnnouncement a;
    a.header.stamp = now();
    a.task_id = t.id;
    a.announcer_id = robot_id_;
    a.position.x = t.x; a.position.y = t.y;
    a.radius = static_cast<float>(t.radius);
    a.confidence = static_cast<float>(t.confidence);
    a.round = t.round;
    a.bid_deadline = t.deadline;
    pub_ann_->publish(a);
    submitBid(t);
  }

  void decide(Task & t, const rclcpp::Time & t_now)
  {
    if (t.bids.empty()) { reannounce(t, t_now); return; }

    // Deterministic across robots GIVEN the same bid set. Tie-break on
    // robot_id so identical utilities cannot split the vote.
    auto best = std::max_element(
      t.bids.begin(), t.bids.end(), [](const M::Bid & a, const M::Bid & b) {
        // NOTE the inversion: max_element wants "a < b", bid_beats says
        // "a wins over b", so the arguments are swapped on purpose.
        return agri_swarm::bid_beats(
          static_cast<double>(b.utility), b.robot_id,
          static_cast<double>(a.utility), a.robot_id);
      });

    if (static_cast<double>(best->utility) <= agri_swarm::kInfeasibleUtility / 10.0) {
      reannounce(t, t_now); return;  // nobody feasible
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
      committed_.push_back(t.id);      // TODO(you): hand to the path executor
    }
  }

  void onAward(const M::TaskAward & aw)
  {
    auto it = tasks_.find(aw.task_id);
    if (it == tasks_.end()) return;
    Task & t = it->second;
    if (aw.round < t.round) return;

    // Disagreement detector. If this fires, your bid window is too short for
    // the DDS latency, or you are dropping messages. LOG IT, do not silence it.
    if (t.phase == Phase::Awarded && !t.winner.empty() && t.winner != aw.winner_id) {
      RCLCPP_WARN(get_logger(),
                  "split-brain on task %u round %u: local winner %s, heard %s "
                  "(%u bids seen there)",
                  aw.task_id, aw.round, t.winner.c_str(),
                  aw.winner_id.c_str(), aw.n_bids_seen);
      split_brain_count_++;
      // Lowest robot_id wins the argument. Arbitrary, but must be consistent.
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

  void reannounce(Task & t, const rclcpp::Time & t_now)
  {
    if (t.round + 1 >= static_cast<uint32_t>(max_rounds_)) {
      RCLCPP_WARN(get_logger(), "task %u abandoned after %d rounds", t.id, max_rounds_);
      done_.insert(t.id);
      t.phase = Phase::Done;
      return;
    }
    t.round++;
    t.bids.clear();
    t.winner.clear();
    t.phase = Phase::Announced;
    t.deadline = t_now + rclcpp::Duration::from_seconds(bid_window_);
    announce(t);
  }

  // ------------------------------------------------------------------- peers

  void onState(const M::RobotState & s)
  {
    last_seen_[s.robot_id] = now();
    if (s.robot_id == robot_id_) {
      energy_j_ = s.remaining_energy_j;
      status_ = s.status;
      if (energy_capacity_j_ <= 0.0) energy_capacity_j_ = s.remaining_energy_j;
      energy_known_ = true;
    }
  }

  bool isSilent(const std::string & id, const rclcpp::Time & t_now) const
  {
    auto it = last_seen_.find(id);
    if (it == last_seen_.end()) return true;
    return (t_now - it->second).seconds() > award_grace_;
  }

  // ------------------------------------------------------------------ state
  std::string robot_id_;
  agri_swarm::BidMode bid_mode_{agri_swarm::BidMode::ConfidenceEnergy};
  double cell_{}, bid_window_{}, award_grace_{};
  int max_rounds_{};
  double energy_per_m_{}, treat_cost_j_{}, reserve_frac_{}, conf_gamma_{};

  double x_{}, y_{};
  bool have_pose_{false};
  double energy_j_{1.0}, energy_capacity_j_{-1.0};
  // False until an energy monitor node exists. While false, the SoC term in
  // utility() is 1.0 and the feasibility gate is a no-op -- BY CONTRACT, and
  // any result produced in this state must say so in the README.
  bool energy_known_{false};
  uint8_t status_{M::RobotState::STATUS_IDLE};
  uint64_t split_brain_count_{0};

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
