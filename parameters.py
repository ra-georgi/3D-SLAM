import numpy as np

default_param_dict = {
  "Initial_State": np.array([4, 4, 2,]),
  "Initial_Covariance": np.zeros((3,3)),
  "Time_Step": 0.1,
  "Final_Time": 20,
  "Quad_Length": 0.2,
  "Landmarks": np.array([             #Coordinates of Landmarks
      # [1, 1, 1],                       #Comment whole line if ignoring landmarks
      [0.5, 0.5, 0.5],
      [2, 7, 8],   
      [7, 5, 5],   
      [3, 6, 4 ],   
      [1, 2, 9],  
      [8, 2, 3],                       
  ]),
  "Sensor_Working_Distance": 6,
}