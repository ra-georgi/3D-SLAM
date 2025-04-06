# UKF SLAM class
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import sqrtm, block_diag

class UKF_SLAM:

    #Constructor
    def __init__(self, param_dict) -> None :

        # Get Data from Input Parameters
        self.mu_0 = param_dict["Initial_State"]
        self.cov_0 = param_dict["Initial_Covariance"]
        self.sim_step_size = param_dict["Time_Step"]
        self.final_time = param_dict["Final_Time"]
        self.quad_length = param_dict["Quad_Length"]
        self.landmark_positions = param_dict["Landmarks"]
        self.sensor_range = param_dict["Sensor_Working_Distance"]
        self.R_control_covariance = param_dict["Control_Covariance"]
        self.Q_sensor_covariance = param_dict["Measurement_Covariance"]
        self.num_landmarks = int(self.landmark_positions.shape[0])

        # UKF Parameters
        self.UKF_alpha = param_dict["UKF_alpha"]
        self.UKF_beta  = param_dict["UKF_beta"]
        self.UKF_kappa = param_dict["UKF_kappa"]

        # Handle Parameters related to simulation time
        self.num_time_steps = int(self.final_time/self.sim_step_size)+1  
        self.final_time =  self.sim_step_size*(self.num_time_steps-1)
        print("")
        print(f"Warning: If final time is not exactly divisible by time step, final time of simulation will be changed")
        print(f"Final Time for current simulation set to {self.final_time}")
        self.time_vector = np.linspace(0, self.final_time, num=self.num_time_steps)

        self.robot_state_size = self.mu_0.size
        self.state_size = (self.robot_state_size) #+ (3*self.num_landmarks)  # 1-D array

        # self.mu = np.zeros((self.state_size,self.num_time_steps))
        self.mu  = [None] * self.num_time_steps

        self.robot_actual_pose = np.zeros((3,self.num_time_steps))

        # Update state/mean vector with initial condition
        self.mu[0] = self.mu_0
        self.robot_actual_pose[:, 0] = self.mu_0

        # Setting up covariance matrix 
        # self.cov =  np.zeros((self.num_time_steps, self.state_size, self.state_size))
        self.cov     = [None] * self.num_time_steps
        self.cov[0]  = np.zeros((self.robot_state_size, self.robot_state_size))
        
        # self.variance_robot_pose = np.zeros((self.robot_state_size, self.robot_state_size))
        # self.variance_landmarks =  1000*np.eye(3*self.num_landmarks)
        # self.cov_robot_with_landmarks =  np.zeros((self.robot_state_size,3*self.num_landmarks))

        # #Should variance_landmarks for future timesteps be initialzed to 1000 too?
        # self.cov[0, :, : ] = np.block([
        #     [self.variance_robot_pose,        self.cov_robot_with_landmarks        ],
        #     [self.cov_robot_with_landmarks.T, self.variance_landmarks              ]
        # ])

        # Each row i has measurements (range, pitch, yaw) corresponding to landmark i
        # 4th column is boolean representing if landmark has been seen yet or not (To increase size of mu and cov)
        # 5th column is boolean representing if landmark was observed this time step (To know which landmarks were observed)
        # 6th column is original landmark number (may not observe in this order)
        self.measurements  =  np.zeros((self.num_landmarks, 6))
        self.landmarks_seen = 0
        # Keys are original landmark numbers, values are current order
        self.landmark_dictionary = dict([  ])   #To relate measurements matrix to current mu vector (as landmarks not observed in order)

    def simulate(self):
        # Simulate Quadcopter motion as per dynamics

        u_odom = np.array([0.2, 0.2, 0.2])
        for i in range(self.num_time_steps-1):

            # Creating Odometry Data
            # u_odom = np.array([0.2*np.sin(self.time_vector[i]), 0.2*np.cos(self.time_vector[i]), 0.2*np.sin(self.time_vector[i])])
            self.robot_actual_pose[:, i+1] = self.robot_actual_pose[:, i] + u_odom
            # u_odom[0] += 0.03*np.random.rand()
            # u_odom[1] += 0.03*np.random.rand()
            # u_odom[2] += 0.03*np.random.rand()
            
            #Prediction Step
            self.mu[i+1], self.cov[i+1] = self.prediction_step(self.mu[i], self.cov[i], u_odom)

            #Updates self.measurements/Get newest measurement
            self.sensor_model(self.robot_actual_pose[:, i+1])  
            
            # #Measurement Update step
            self.mu[i+1], self.cov[i+1] = self.measurement_update(self.mu[i+1], self.cov[i+1])

            if ( max(self.mu[i+1]) > 1000000 ):
                    self.time_vector = 0
                    print("Numerical issues")
                    break
            
    def prediction_step(self, mu_previous, cov_previous, u_odom):
        #Perform prediction step of UKF based on odometry data

        #Don't do mu_current =  mu_previous, will overwrite
        mu_current =  mu_previous.copy()
        cov_current = cov_previous.copy()

        self.state_size = mu_current.shape[0]
        self.UKF_lambda = (self.UKF_alpha*self.UKF_alpha*(self.state_size+self.UKF_kappa))-self.state_size

        # Compute original distribution's sigma points
        sigma_points, weights_mean, weights_covariance = self.calc_sigma_points(mu_current, cov_current)
        sigma_points += u_odom.reshape((3,1))

        # Recover mu and sigma of the transformed distribution
        mu_current, cov_current = self.recover_gaussian(sigma_points, weights_mean, weights_covariance)

        cov_current[0,0] += self.R_control_covariance
        cov_current[1,1] += self.R_control_covariance
        cov_current[2,2] += self.R_control_covariance

        return mu_current, cov_current
    
    def calc_sigma_points(self, mu_current, cov_current):

        n = self.state_size
        sigma_points = np.zeros((n, 2*n+1))
        sqrt_cov = sqrtm((n+self.UKF_lambda)*cov_current)
    
        sigma_points[:,0] = mu_current
        # Check this, check everything up to this point
        for i in range(1,n+1):
            sigma_points[:,i] = mu_current + sqrt_cov[:,i-1]
        for i in range(n+1,(2*n)+1):
            sigma_points[:,i] = mu_current - sqrt_cov[:,(i-1)-n]

        weights_mean = np.zeros((2*n)+1)
        weights_covariance = np.zeros((2*n)+1)

        weights_mean[0] = self.UKF_lambda / (n + self.UKF_lambda)
        weights_covariance[0] = weights_mean[0] + (1 - (self.UKF_alpha**2) + self.UKF_beta)

        weights_mean[1:]       = 1 / (2 * (n + self.UKF_lambda))
        weights_covariance[1:] = 1 / (2 * (n + self.UKF_lambda))

        return sigma_points, weights_mean, weights_covariance
    
    def recover_gaussian(self, sigma_points, weights_mean, weights_covariance):
        # Correct in prediction, only robot sigma points should change

        self.state_size = sigma_points.shape[0]
        mu  = np.zeros(self.state_size)
        cov = np.zeros((self.state_size,self.state_size))

        for i in range(sigma_points.shape[1]):
            mu  = mu   + (weights_mean[i]*sigma_points[:, i])

        for i in range(sigma_points.shape[1]):
            deviation = sigma_points[:, i]-mu
            cov = cov  + (weights_covariance[i]*(np.outer(deviation,deviation)))

        return mu, cov
    
    def sensor_model(self, mu_estimate):
        #Simulates range, pitch, and yaw sensor
        
        self.measurements[:, 4] = 0  #Reset observations tracker
        landmark_rel_position =  1e5*np.ones((self.num_landmarks,3))
        landmark_range        =  1e5*np.ones((self.num_landmarks))

        for i in range(self.num_landmarks):
            self.measurements[i, 5] = i                        #Suboptimal I know, can initialize correctly later. Also starting from 0
            landmark_rel_position[i,:] = self.landmark_positions[i, :] - mu_estimate[0:3]
            landmark_range[i] = np.sqrt( np.dot(landmark_rel_position[i,:],landmark_rel_position[i,:]) )

            if landmark_range[i] < self.sensor_range:
                # self.measurements[i, 3] = 1             #Set in measurment update step
                self.measurements[i, 4] = 1

                del_x = landmark_rel_position[i,0]
                del_y = landmark_rel_position[i,1]
                del_z = landmark_rel_position[i,2]
                xy_distance = np.sqrt( (del_x**2) + (del_y**2))

                self.measurements[i,0] = landmark_range[i] #+ 0.03*np.random.rand()
                self.measurements[i,1] = np.arctan2(del_z, xy_distance) #+ 0.03*np.random.rand()
                self.measurements[i,2] = np.arctan2(del_y, del_x)  #+ 0.03*np.random.rand()

    def measurement_update(self, mu_prediction, cov_prediction):
        # Measurement incorporation step of Kalman filter

        x_landmark = y_landmark = z_landmark = xy = 0
        x_robot = y_robot = z_robot = 0
        mu =  mu_prediction.copy()
        cov = cov_prediction.copy()
        
        measurement = np.array([])
        expected_measurement =  np.array([])
        #Number of measurements in this time step
        current_measurement = self.measurements[self.measurements[:, 4] == 1]

        # H = np.zeros((3*num_measurements,self.state_size))
        # measurement_num  = 0   #To properly create H matrix as not all measurment contribute to H at every time step

        for measurement_index in range(current_measurement.shape[0]):

            range_reading = current_measurement[measurement_index, 0]
            pitch_reading = current_measurement[measurement_index, 1] 
            yaw_reading   = current_measurement[measurement_index, 2] 

            original_landmark_num = current_measurement[measurement_index, 5].astype(np.int64)
                
            #Initialize Landmark position if not previously observed
            if (current_measurement[measurement_index, 3] == 0):

                self.measurements[original_landmark_num, 3] = 1  #Marking as observed
                self.landmark_dictionary[original_landmark_num] = self.landmarks_seen
                self.landmarks_seen += 1

                z_landmark = range_reading*np.sin(pitch_reading)
                xy = range_reading*np.cos(pitch_reading)
                y_landmark  = xy*np.sin(yaw_reading)
                x_landmark =  xy*np.cos(yaw_reading)
                x_robot = mu[0]
                y_robot = mu[1]
                z_robot = mu[2]
                mu = np.append(mu, [x_robot+x_landmark, y_robot+y_landmark, z_robot+z_landmark])
                cov = block_diag(cov, self.Q_sensor_covariance*np.eye(3) )
                
                    
        #     landmark_rel_position = mu[self.robot_state_size+(3*i) : self.robot_state_size+(3*i)+3] - mu[0:3]
        #     del_x = landmark_rel_position[0]
        #     del_y = landmark_rel_position[1]
        #     del_z = landmark_rel_position[2]
        #     xy_distance = np.sqrt( (del_x**2) + (del_y**2))

        #     predicted_range = np.sqrt( np.dot(landmark_rel_position,landmark_rel_position) )
        #     predicted_pitch = np.arctan2(del_z, xy_distance)
        #     predicted_yaw   = np.arctan2(del_y, del_x)

        #     measurement = np.append(measurement, self.measurements[i, 0:3])
        #     expected_measurement = np.append(expected_measurement, [predicted_range,predicted_pitch, predicted_yaw])

        #     measurement_num  += 1


        # K = (cov @ (H.T)) @ np.linalg.inv( (H @ cov @ (H.T)) + Q)

        # mu = mu + (K @ (measurement-expected_measurement))
        # cov = (np.eye(self.state_size) - (K @ H) ) @ cov;        
        
        return mu, cov