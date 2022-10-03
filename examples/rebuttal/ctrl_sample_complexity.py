from schrl.rebuttal.control_sample_complexity.goal_conditioned_policy import train_point_gc_policy
from schrl.rebuttal.control_sample_complexity.options import train_point_option_policy

if __name__ == '__main__':
    train_point_gc_policy()
    train_point_option_policy("north")
    train_point_option_policy("south")
    train_point_option_policy("east")
    train_point_option_policy("west")
