# A single weed patch reported by a robot's detector.
# Confidence is the detector's BELIEF, not ground truth. The project hinges on
# this being noisy, so it must never be filled from the oracle CSV.

std_msgs/Header header

uint32   patch_id          # oracle id; for OFFLINE SCORING ONLY, never for bidding
string   observer_id
geometry_msgs/Point position
float32  radius            # estimated patch radius [m]
float32  confidence        # [0,1]
