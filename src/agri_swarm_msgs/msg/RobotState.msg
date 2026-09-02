# Heartbeat. Absence for > state_timeout is how failure is detected;
# the fault injector simply stops publishing this.

std_msgs/Header header

uint8 STATUS_IDLE=0
uint8 STATUS_TRANSIT=1
uint8 STATUS_TREATING=2
uint8 STATUS_LOW_BATTERY=3
uint8 STATUS_RETURNING=4
uint8 STATUS_FAILED=5

string   robot_id
geometry_msgs/Pose pose
float32  remaining_energy_j
float32  estimated_range_m
uint8    status
uint32[] committed_task_ids
