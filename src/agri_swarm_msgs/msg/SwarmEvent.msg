# One row per OBSERVER, not per event. De-duplicate on (event_type, task_id, round).
uint8 EVENT_REANNOUNCE=0    # a round failed and the task went back to auction
uint8 EVENT_ABANDON=1       # max_rounds exhausted; the task is dropped
uint8 EVENT_SPLIT_BRAIN=2   # two awards for the same task named different winners

std_msgs/Header header
uint8 event_type
string observer_id          # the allocator that emitted this message

uint32 task_id
uint32 round                # the observer's round for this task
geometry_msgs/Point position
float32 confidence

string winner_a             # the observer's local winner, empty if none
string winner_b             # the conflicting announced winner (split-brain only)
uint32 n_bids_a             # bids the observer had collected
uint32 n_bids_b             # bids the conflicting announcer reported

string detail