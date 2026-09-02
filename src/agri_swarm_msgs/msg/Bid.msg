# One robot's bid on one task, for one round. Higher utility wins.
# Components are carried separately so the ablation (distance-only vs
# confidence x energy) becomes a config flag, not a code change.

std_msgs/Header header

uint32   task_id
uint32   round
string   robot_id

float32  utility           # the number that actually competes
float32  travel_cost_m
float32  energy_cost_j
float32  remaining_energy_j
bool     feasible          # false = cannot reach it and still return within budget
