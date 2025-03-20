#main file

# To Do:
#If updating state with quaternions/angles, remember to normalize the angles

from EKF_SLAM import EKF_SLAM
from parameters import default_param_dict

#Create SLAM object
SLAM_Object = EKF_SLAM(default_param_dict)
# print(SLAM_Object.__dict__)

SLAM_Object.simulate()
# SLAM_Object.animate_quad()
# print(SLAM_Object.mu)


