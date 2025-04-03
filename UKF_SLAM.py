# UKF SLAM class
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import sqrtm

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
        self.mu = np.zeros((self.state_size,self.num_time_steps))
        self.robot_actual_pose = np.zeros((3,self.num_time_steps))

        # Update state/mean vector with initial condition
        self.mu[0:self.robot_state_size, 0] = self.mu_0
        self.robot_actual_pose[:, 0] = self.mu_0

        # Setting up covariance matrix 
        self.cov =  np.zeros((self.num_time_steps, self.state_size, self.state_size))
        
        # self.variance_robot_pose = np.zeros((self.robot_state_size, self.robot_state_size))
        # self.variance_landmarks =  1000*np.eye(3*self.num_landmarks)
        # self.cov_robot_with_landmarks =  np.zeros((self.robot_state_size,3*self.num_landmarks))

        # #Should variance_landmarks for future timesteps be initialzed to 1000 too?
        # self.cov[0, :, : ] = np.block([
        #     [self.variance_robot_pose,        self.cov_robot_with_landmarks        ],
        #     [self.cov_robot_with_landmarks.T, self.variance_landmarks              ]
        # ])

        # Each row i has measurements (range, pitch, yaw) corresponding to landmark i
        # 4th column is boolean representing if landmark has been seen yet or not (To increase size of mu and cov, not required now)
        # 5th column is boolean representing if landmark was observed this time step (To know which landmarks were observed)
        self.measurements  =  np.zeros((self.num_landmarks, 5))

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
            self.mu[:,i+1], self.cov[i+1,:,:] = self.prediction_step(self.mu[:,i], self.cov[i,:,:,], u_odom)

            # #Updates self.measurements/Get newest measurement
            # self.sensor_model(self.robot_actual_pose[:, i+1])  
            
            # #Measurement Update step
            # self.mu[:,i+1], self.cov[i+1,:,:] = self.measurement_update(self.mu[:,i+1], self.cov[i+1,:,:])

            if ( max(self.mu[:,i+1]) > 1000000 ):
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