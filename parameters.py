import numpy as np

default_param_dict = {
  "Initial_State": np.array([0, 0, 0, 0, 0, 0]),
  "Initial_Covariance": np.zeros((6,6)),
  "Time_Step": 0.5,
  "Final_Time": 2,
  "Quad_Length": 0.2,
}