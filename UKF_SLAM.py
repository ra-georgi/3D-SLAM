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
        self.measurements[:,5]  =  -1
        self.landmarks_seen = 0
        # Keys are original landmark numbers, values are current order
        # self.landmark_dictionary = dict([  ])   #To relate measurements matrix to current mu vector (as landmarks not observed in order)

    def simulate(self):
        # Simulate Quadcopter motion as per dynamics

        # u_odom = np.array([0.1, 0.1, 0.1])
        for i in range(self.num_time_steps-1):

            # Creating Odometry Data
            u_odom = np.array([0.2*np.sin(self.time_vector[i]), 0.2*np.cos(self.time_vector[i]), 0.2*np.sin(self.time_vector[i])])
            self.robot_actual_pose[:, i+1] = self.robot_actual_pose[:, i] + u_odom
            u_odom[0] += 0.03*np.random.rand()
            u_odom[1] += 0.03*np.random.rand()
            u_odom[2] += 0.03*np.random.rand()
            
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
        sigma_points[0:3,:] += u_odom.reshape((3,1))

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
            landmark_rel_position[i,:] = self.landmark_positions[i, :] - mu_estimate[0:3]
            landmark_range[i] = np.sqrt( np.dot(landmark_rel_position[i,:],landmark_rel_position[i,:]) )

            if landmark_range[i] < self.sensor_range:
                # self.measurements[i, 3] = 1             #Set in measurment update step
                self.measurements[i, 4] = 1

                del_x = landmark_rel_position[i,0]
                del_y = landmark_rel_position[i,1]
                del_z = landmark_rel_position[i,2]
                xy_distance = np.sqrt( (del_x**2) + (del_y**2))

                self.measurements[i,0] = landmark_range[i] + 0.03*np.random.rand()
                self.measurements[i,1] = np.arctan2(del_z, xy_distance) + 0.03*np.random.rand()
                self.measurements[i,2] = np.arctan2(del_y, del_x)  + 0.03*np.random.rand()

    def measurement_update(self, mu_prediction, cov_prediction):
        # Measurement incorporation step of Kalman filter

        x_landmark = y_landmark = z_landmark = xy = 0
        x_robot = y_robot = z_robot = 0
        mu =  mu_prediction.copy()
        cov = cov_prediction.copy()
        
        #Number of measurements in this time step
        num_measurements = int(sum(self.measurements[:,4]))
        landmark_indexes_current = []
        measurement = np.array([])
        mean_expected_measurement =  np.zeros(3*num_measurements)


        for measurement_index in range(self.num_landmarks):

            # Not Measured in this time step             
            if (int(self.measurements[measurement_index, 4]) == 0):
                continue

            range_reading = self.measurements[measurement_index, 0]
            pitch_reading = self.measurements[measurement_index, 1] 
            yaw_reading   = self.measurements[measurement_index, 2] 

            #Initialize Landmark position if not previously observed
            if (self.measurements[measurement_index, 3] == 0):

                self.measurements[measurement_index, 3] = 1  #Marking as observed
                self.measurements[measurement_index, 5] = self.landmarks_seen
                self.landmarks_seen += 1
                self.state_size += 3

                z_landmark = range_reading*np.sin(pitch_reading)
                xy = range_reading*np.cos(pitch_reading)
                y_landmark  = xy*np.sin(yaw_reading)
                x_landmark =  xy*np.cos(yaw_reading)
                x_robot = mu[0]
                y_robot = mu[1]
                z_robot = mu[2]
                mu = np.append(mu, [x_robot+x_landmark, y_robot+y_landmark, z_robot+z_landmark])
                cov = block_diag(cov, self.Q_sensor_covariance*np.eye(3) )  # Not as per formula, but I'm trying it

            # measurement = np.append(measurement, self.measurements[measurement_index, 0:3])  
            landmark_indexes_current.append(int(self.measurements[measurement_index, 5]))


        # Compute original distribution's sigma points
        sigma_points, weights_mean, weights_covariance = self.calc_sigma_points(mu, cov)

        num_sigma_points = sigma_points.shape[1]
        sigma_expected_measurement = np.zeros((3*num_measurements, num_sigma_points))
        
        for i in range(num_sigma_points):
            sigma_expected_measurement[:,i], measurement = self.calc_expected_measurement(sigma_points[:,i],landmark_indexes_current)
        
        for i in range(num_sigma_points):
            mean_expected_measurement  = mean_expected_measurement   + (weights_mean[i]*sigma_expected_measurement[:,i])

        St = np.zeros((3*num_measurements, 3*num_measurements))
        cross_cov = np.zeros((self.state_size, 3*num_measurements))

        for i in range(num_sigma_points):
            deviation_measurement = sigma_expected_measurement[:,i] - mean_expected_measurement
            deviation_mean        = sigma_points[:, i] - mu
            St  = St   + (weights_covariance[i]*(np.outer(deviation_measurement,deviation_measurement)))
            cross_cov  = cross_cov   + (weights_covariance[i]*(np.outer(deviation_mean,deviation_measurement)))

        St += self.Q_sensor_covariance*np.eye(3*num_measurements)

        K = (cross_cov) @ np.linalg.inv(St)

        mu = mu + (K @ (measurement-mean_expected_measurement))
        cov = cov - (K@St@ (K.T))                           #(np.eye(self.state_size) - (K @ H) ) @ cov;        

        
        return mu, cov
    

    def calc_expected_measurement(self, mu_estimate, landmark_indexes_current):
        #Simulates range, pitch, and yaw sensor
        
        measurement = np.array([])
        expected_measurement =  np.array([])

        current_measurement  =  self.measurements[self.measurements[:,4] == 1]
        order = np.argsort(current_measurement[:, 5])
        reordered_arr = current_measurement[order]

        for idx in range(reordered_arr.shape[0]):
            i = int(reordered_arr[idx,5])
            landmark_rel_position = mu_estimate[self.robot_state_size+(3*i) : self.robot_state_size+(3*i)+3] - mu_estimate[0:3]
            del_x = landmark_rel_position[0]
            del_y = landmark_rel_position[1]
            del_z = landmark_rel_position[2]
            xy_distance = np.sqrt( (del_x**2) + (del_y**2))

            predicted_range = np.sqrt( np.dot(landmark_rel_position,landmark_rel_position) )
            predicted_pitch = np.arctan2(del_z, xy_distance)
            predicted_yaw   = np.arctan2(del_y, del_x)
            measurement =     np.append(measurement, reordered_arr[idx, 0:3])
            expected_measurement = np.append(expected_measurement, [predicted_range, predicted_pitch, predicted_yaw])        

        return expected_measurement, measurement

    def animate_quad(self):

        fig = plt.figure(figsize=(8,6))
        self.ax_anim = plt.axes(projection='3d')
        # Adjust the size of the plot within the figure
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self.ax2 = plt.axes([0.8, 0.7, 0.2, 0.2])
        self.ax2.axis('off')  # Hide the axes for the inset      

        # Initialize the text in the inset axes with a box around it
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
        self.text_t = self.ax2.text(0.1, 0.8, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)
        self.text_x = self.ax2.text(0.1, 0.6, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)   
        self.text_y = self.ax2.text(0.1, 0.4, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)
        self.text_z = self.ax2.text(0.1, 0.2, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)                                             
        
        x0, y0, z0 = self.robot_actual_pose[0,0], self.robot_actual_pose[1,0], self.robot_actual_pose[2,0]

        l = self.quad_length
        self.quad_Arm1 = self.ax_anim.plot3D([x0+l, x0-l], [y0, y0], [z0, z0], lw=3 )[0]
        self.quad_Arm2 = self.ax_anim.plot3D([x0, x0], [y0+l, y0-l], [z0, z0], lw=3 )[0]
        self.quad_traj = self.ax_anim.plot3D(x0, y0, z0, 'gray')[0] 

        #To make quadcopter's arms look equal in animation
        self.ax_anim.set_xlim([0,10])
        self.ax_anim.set_ylim([0,10])
        self.ax_anim.set_zlim([0,10])      
  
        r = 0.1
        #Plot Landmarks
        u, v = np.mgrid[0:2*np.pi:7j, 0:np.pi:7j]
        x_LM = r*np.cos(u)*np.sin(v)
        y_LM = r*np.sin(u)*np.sin(v) 
        z_LM = r*np.cos(v)
        for obs in self.landmark_positions:
                self.ax_anim.plot_surface(x_LM+obs[0], y_LM+obs[1], z_LM+obs[2], color='b',alpha=0.3)       
        
        # Initialize the ellipsoid surface
        x_robot, y_robot, z_robot = self.get_ellipsoid(self.mu[0][0:3] , self.cov[0][ 0:3, 0:3])    
        self.surf = self.ax_anim.plot_surface(x_robot, y_robot, z_robot, color='b', alpha=0.3)

        self.surf_landmarks = [None] * self.num_landmarks
        self.surf_observations = [None] * self.num_landmarks

        self.ani = FuncAnimation(fig=fig, func=self.update_animation_frame,frames=self.num_time_steps, fargs=(self,),interval=50)
        plt.show()

    @staticmethod
    def update_animation_frame(frame,self):
            # for each frame, update the data stored on each artist.

            time = self.time_vector[frame]
            xt, yt, zt = self.robot_actual_pose[0,frame], self.robot_actual_pose[1,frame], self.robot_actual_pose[2,frame]

            Q = np.eye(3)

            l = self.quad_length

            Arm1_Start = np.array([l,0,0])
            Arm1_End = np.array([-l,0,0])

            Arm2_Start = np.array([0,l,0])
            Arm2_End = np.array([0,-l,0])

            Arm1_Start = Q @ Arm1_Start
            Arm1_End = Q @ Arm1_End
            Arm2_Start = Q @ Arm2_Start
            Arm2_End = Q @ Arm2_End              

            self.quad_Arm1.set_data_3d([xt+Arm1_Start[0], xt+Arm1_End[0]], [yt+Arm1_Start[1], yt+Arm1_End[1]], [zt+Arm1_Start[2], zt+Arm1_End[2]])
            self.quad_Arm2.set_data_3d([xt+Arm2_Start[0], xt+Arm2_End[0]], [yt+Arm2_Start[1], yt+Arm2_End[1]], [zt+Arm2_Start[2], zt+Arm2_End[2]])
            self.quad_traj.set_data_3d(self.robot_actual_pose[0,:frame],self.robot_actual_pose[1,:frame],self.robot_actual_pose[2,:frame])

            self.text_t.set_text(f't = {time:.2f} s')
            self.text_x.set_text(f'x = {xt:.2f} m')
            self.text_y.set_text(f'y = {yt:.2f} m')
            self.text_z.set_text(f'z = {zt:.2f} m')

            self.surf.remove()  # Remove the old surface
            x_robot, y_robot, z_robot = self.get_ellipsoid(self.mu[frame][0:3], self.cov[frame][0:3, 0:3])    
            self.surf = self.ax_anim.plot_surface(x_robot, y_robot, z_robot, color='b', alpha=0.2)

            for i in range(self.num_landmarks):
                if self.surf_landmarks[i] is not None:
                    temp_surf = self.surf_landmarks[i]
                    temp_surf.remove()  # Initial covariance is set to a high value, so will cover entire plot  

                landmark_rel_order = int(self.measurements[i, 5])
                dim_ss = self.mu[frame].shape[0]
                landmark_rel_index = self.robot_state_size + (3*landmark_rel_order)
                if dim_ss > (landmark_rel_index + 1):
                    # Check covariance size (using max eigenvalue)
                    # landmark_start_index = self.robot_state_size + (3*i)
                    eigvals = np.linalg.eigh(self.cov[frame][landmark_rel_index : landmark_rel_index + 3, landmark_rel_index : landmark_rel_index + 3])[0]  # Only need eigenvalues
                    if np.max(eigvals) < 1000:  # Plot only if covariance is not too large
                        x_LM, y_LM, z_LM = self.get_ellipsoid(
                                self.mu[frame][landmark_rel_index : landmark_rel_index + 3], 
                                self.cov[frame][landmark_rel_index : landmark_rel_index + 3, landmark_rel_index : landmark_rel_index + 3]
                                )
                        self.surf_landmarks[i] = self.ax_anim.plot_surface(x_LM, y_LM, z_LM , color='r', alpha=0.3)   
                    else:
                        self.surf_landmarks[i] = None

            for i in range(self.num_landmarks):
                landmark_rel_position =  1e5*np.ones((self.num_landmarks,3))
                landmark_range        =  1e5*np.ones((self.num_landmarks))
                if self.surf_observations[i] is not None:
                    temp_surf = self.surf_observations[i]
                    temp_surf.remove()  
                landmark_rel_position[i,:] = self.landmark_positions[i, :] - self.mu[frame][0:3]
                landmark_range[i] = np.sqrt( np.dot(landmark_rel_position[i,:],landmark_rel_position[i,:]) )
                if landmark_range[i] < self.sensor_range:
                    self.surf_observations[i] = self.ax_anim.plot3D(
                         [self.mu[frame][0], self.landmark_positions[i, 0]], 
                         [self.mu[frame][1], self.landmark_positions[i, 1]], 
                         [self.mu[frame][2], self.landmark_positions[i, 2]], lw=3 )[0]
                else:                 
                    self.surf_observations[i] = None

            
            return 


    def get_ellipsoid(self, mean, cov, scale=2.45):  # scale=2.45 for ~95% confidence
         
        # Function to generate ellipsoid points
        u = np.linspace(0, 2 * np.pi, 15)  # Reduced resolution for speed
        v = np.linspace(0, np.pi, 15)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        sphere = np.stack([x, y, z], axis=0) * scale
        
        eigvals, eigvecs = np.linalg.eigh(cov)
        radii = np.sqrt(np.maximum(eigvals, 0))  # Ensure non-negative
        transform = eigvecs @ np.diag(radii)
        
        ellipsoid = transform @ sphere.reshape(3, -1) + mean[:, np.newaxis]
        return (ellipsoid[0].reshape(x.shape),
                ellipsoid[1].reshape(y.shape),
                ellipsoid[2].reshape(z.shape))    