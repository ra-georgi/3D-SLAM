import numpy as np

default_param_dict = {
  "Initial_State": np.array([1, 1, 1,]),
  "Initial_Covariance": np.zeros((3,3)),
  "Time_Step": 0.05,
  "Final_Time": 30,
  "Quad_Length": 0.2,
  "Landmarks": np.array([             #Coordinates of Landmarks
      # [1, 1, 1],                       #Comment whole line if ignoring landmarks
      [0.5, 0.5, 0.5],
      [2, 3, 4],
      # [2, 7, 8],   
      # [7, 5, 5],   
      # [3, 6, 4 ],   
      # [1, 2, 8],  
      # [8, 2, 3], 
      # [7, 7 ,7],                      
  ]),
  "Sensor_Working_Distance": 3,
  "Control_Covariance": 0.1,
  "Measurement_Covariance": 0.1,
  "UKF_alpha" : 0.9,
  "UKF_beta"  : 2,
  "UKF_kappa" : 1,
}