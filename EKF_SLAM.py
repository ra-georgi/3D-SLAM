# EKF SLAM class

# Target EKF SLAM
# Create virtual environment

# Step 1: Prediction
    # Use Odometry
    # Use Control Commands
# Step 2: Measurment Update

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EKF_SLAM:

    #Constructor
    def __init__(self, param_dict) -> None :
        self.mu_0 = param_dict["Initial_State"]
        self.cov_0 = param_dict["Initial_Covariance"]
        self.sim_step_size = param_dict["Time_Step"]
        self.final_time = param_dict["Final_Time"]
        self.quad_length = param_dict["Quad_Length"]
        self.landmark_positions = param_dict["Landmarks"]
        self.sensor_range = param_dict["Sensor_Working_Distance"]

        self.num_time_steps = int(self.final_time/self.sim_step_size)+1  

        # Below logic doesn't work with decimal value, can retry by multiplying by 10's and checking
        # if (self.final_time%num_time_steps) != 0:
        #        self.final_time =  self.sim_step_size*(num_time_steps-1)
        #        print(f"Final time not exactly divisible by time step, final time of simulation changed to {self.final_time}")

        self.final_time =  self.sim_step_size*(self.num_time_steps-1)
        print("")
        print(f"Warning: If final time is not exactly divisible by time step, final time of simulation will be changed")
        print(f"Final Time for current simulation set to {self.final_time}")

        self.time_vector = np.linspace(0, self.final_time, num=self.num_time_steps)

        self.num_landmarks = int(self.landmark_positions.shape[0])
        self.robot_state_size = self.mu_0.size

        self.state_size = (self.robot_state_size) + (3*self.num_landmarks)  # 1-D array
        # cov_mat_dim = int(self.cov_0.shape[0])
        
        self.mu = np.zeros((self.state_size,self.num_time_steps))
        # Update state/mean vector with initial condition
        self.mu[0:self.robot_state_size, 0] = self.mu_0

        self.cov =  np.zeros((self.state_size, self.state_size, self.num_time_steps))

        self.variance_robot_pose = np.zeros((self.robot_state_size, self.robot_state_size))
        self.variance_landmarks =  1000*np.eye(3*self.num_landmarks)
        self.cov_robot_with_landmarks =  np.zeros((self.robot_state_size,3*self.num_landmarks))

        #Should variance_landmarks for future timesteps be initialzed to 1000 too?
        self.cov[:,:, 0] = np.block([
            [self.variance_robot_pose,        self.cov_robot_with_landmarks        ],
            [self.cov_robot_with_landmarks.T, self.variance_landmarks              ]
        ])

        # Each row i has measurements (range, pitch, yaw) corresponding to landmark i
        # 4th column is boolean representing if landmark has been seen yet or not (To increase size of mu and cov, not required now)
        # 5th column is boolean representing if landmark was observed this time step (To know which landmarks were observed)
        self.measurements  =  np.zeros((self.num_landmarks, 5))

    def simulate(self):
        # Simulate Quadcopter motion as per dynamics
        u_odom = np.array([0.2, 0.2, 0.2])
        for i in range(self.num_time_steps-1):
            # self.mu[0:3,i+1] = self.mu[0:3,i] + 0.5

            #Prediction Step
            self.mu[:,i+1], self.cov[:,:,i+1] = self.prediction_step(self.mu[:,i], self.cov[:,:,i], u_odom)

            #Updates self.measurements/Get newest measurement
            self.sensor_model(self.mu[:,i+1])  
            
            #Measurement Update step
            self.measurement_update(self.mu[:,i+1], self.cov[:,:,i+1])

            if ( max(self.mu[:,i+1]) > 1000000 ):
                    self.time_vector = 0
                    print("Numerical issues")
                    break
            
    def prediction_step(self, mu_previous, cov_previous, u_odom):
         #Perform prediction step of EKF based on odometry data

        #Don't do mu_current =  mu_previous, will overwrite
        mu_current =  mu_previous.copy()
        cov_current = cov_previous.copy()

        mu_current[0:3] += u_odom
        R = 0.05
        cov_current[0,0] += R
        cov_current[1,1] += R
        cov_current[2,2] += R

        return mu_current, cov_current

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

                self.measurements[i,0] = landmark_range[i]
                self.measurements[i,1] = np.arctan2(del_z, xy_distance)
                self.measurements[i,2] = np.arctan2(del_y, del_x)

    def measurement_update(self, mu, cov):
        # Measurement incorporation step of Kalman filter

        x_landmark = y_landmark = z_landmark = xy = 0
        x_robot = y_robot = z_robot = 0
        
        measurement = np.array([])
        expected_measurement =  np.array([])
        #Number of measurements in this time step
        num_measurements = int(sum(self.measurements[:,4]))
        H = np.zeros((3*num_measurements,self.state_size))
        measurment_num  = 0   #To properly create H matrix as not all measurment contribute to H at every time step

        for i in range(self.num_landmarks):

            #Measured in this time step             
            if (self.measurements[i, 4] == 1):
                range_reading = self.measurements[i, 0]
                pitch_reading = self.measurements[i, 1] 
                yaw_reading   = self.measurements[i, 2] 
                
                #Initialize Landmark position if not previously observed
                if (self.measurements[i, 3] == 0):
                    self.measurements[i, 3] = 1
                    z_landmark = range_reading*np.sin(pitch_reading)
                    xy = range_reading*np.cos(pitch_reading)
                    y_landmark  = xy*np.sin(yaw_reading)
                    x_landmark =  xy*np.cos(yaw_reading)
                    x_robot = mu[0]
                    y_robot = mu[1]
                    z_robot = mu[2]
                    mu[self.robot_state_size+(3*i) : self.robot_state_size+(3*i)+3] = [x_robot+x_landmark, y_robot+y_landmark, z_robot+z_landmark]
                     
                landmark_rel_position = mu[self.robot_state_size+(3*i) : self.robot_state_size+(3*i)+3] - mu[0:3]
                del_x = landmark_rel_position[0]
                del_y = landmark_rel_position[1]
                del_z = landmark_rel_position[2]
                xy_distance = np.sqrt( (del_x**2) + (del_y**2))

                predicted_range = np.sqrt( np.dot(landmark_rel_position,landmark_rel_position) )
                predicted_pitch = np.arctan2(del_z, xy_distance)
                predicted_yaw   = np.arctan2(del_y, del_x)

                measurement = np.append(measurement, self.measurements[i, 0:3])
                expected_measurement = np.append(expected_measurement, [predicted_range,predicted_pitch, predicted_yaw])
                # print(f"Relative Coords: {[del_x,del_x,del_z]}")

                jacobian_range_with_robot_pose = (-1/predicted_range)*np.array([del_x, del_y, del_z])
                jacobian_range_with_landmark = (1/predicted_range)*np.array([del_x, del_y, del_z])
                H[ (3*measurment_num) , 0:3 ] = jacobian_range_with_robot_pose
                H[ (3*measurment_num) , self.robot_state_size+(3*i) : self.robot_state_size+(3*i)+3] = jacobian_range_with_robot_pose

                measurment_num  += 1

        print(f"Measurement: {measurement}")
        print(f"H: {H}")
        print("")
        
                


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
        
        x0, y0, z0 = self.mu[0,0], self.mu[1,0], self.mu[2,0]

        l = self.quad_length
        self.quad_Arm1 = self.ax_anim.plot3D([x0+l, x0-l], [y0, y0], [z0, z0], lw=3 )[0]
        self.quad_Arm2 = self.ax_anim.plot3D([x0, x0], [y0+l, y0-l], [z0, z0], lw=3 )[0]
        self.quad_traj = self.ax_anim.plot3D(x0, y0, z0, 'gray')[0] 

        #To make quadcopter's arms look equal in animation
        self.ax_anim.set_xlim([0,10])
        self.ax_anim.set_ylim([0,10])
        self.ax_anim.set_zlim([0,10])      
  
        # if self.CA:
        #         r = 0.5
        #         #Plot obstacles
        #         u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        #         x = r*np.cos(u)*np.sin(v)
        #         y = r*np.sin(u)*np.sin(v) 
        #         z = r*np.cos(v)
        #         for obs in self.obs_coords:
        #                 self.ax_anim.plot_surface(x+obs[0], y+obs[1], z+obs[2], color='b',alpha=0.3)       
        # 
 
        self.ani = FuncAnimation(fig=fig, func=self.update_animation_frame,frames=self.num_time_steps, fargs=(self,),interval=45)
        plt.show()
    
    @staticmethod
    def update_animation_frame(frame,self):
            # for each frame, update the data stored on each artist.

            time = self.time_vector[frame]
            xt, yt, zt = self.mu[0,frame], self.mu[1,frame], self.mu[2,frame]

            # quat = np.array(self.mu[3:7,frame]).reshape(4,1)
            # Q = self.quat_to_rotmat(quat)
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
            self.quad_traj.set_data_3d(self.mu[0,:frame],self.mu[1,:frame],self.mu[2,:frame])

            self.text_t.set_text(f't = {time:.2f} s')
            self.text_x.set_text(f'x = {xt:.2f} m')
            self.text_y.set_text(f'y = {yt:.2f} m')
            self.text_z.set_text(f'z = {zt:.2f} m')
            return 


      
    
