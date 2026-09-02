# Broadcast when a robot wants a weed patch treated by whoever bids best.
# Any robot may announce; there is no broker.

std_msgs/Header header

uint32   task_id           # quantized-position hash, so duplicate sightings collide
string   announcer_id
geometry_msgs/Point position
float32  radius
float32  confidence        # fused confidence known to the announcer
uint32   round             # increments on re-announcement after a silent winner
builtin_interfaces/Time bid_deadline
