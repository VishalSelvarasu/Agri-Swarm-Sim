# Published by the robot that believes it won, as a commitment.
# Every robot computes the winner independently from the bids it received.
# This message exists to DETECT disagreement (split-brain), not to decide it.

std_msgs/Header header

uint32   task_id
uint32   round
string   winner_id
float32  winning_utility
uint8    n_bids_seen       # bids the winner counted; mismatch => message loss
