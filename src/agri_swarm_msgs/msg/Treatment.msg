# One herbicide application, emitted by the robot that performed it.
# The logger writes one CSV row per message and performs no aggregation.

std_msgs/Header header

uint32   task_id
string   robot_id
geometry_msgs/Point position   # world position of the weed, not of the robot
float32  confidence            # announced confidence at the time of treatment