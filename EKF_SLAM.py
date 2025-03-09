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
        self.Quad_Length = param_dict["Quad_Length"]

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

        state_size = self.mu_0.size  # 1-D array
        cov_mat_dim = int(self.cov_0.shape[0])
        
        self.mu = np.zeros((state_size,self.num_time_steps))
        self.cov =  np.zeros((cov_mat_dim, cov_mat_dim, self.num_time_steps))

    def simulate(self):
        # Simulate Quadcopter motion as per dynamics
        for i in range(self.num_time_steps-1):
            self.mu[0:3,i+1] = self.mu[0:3,i] + 0.5
            if ( max(self.mu[:,i+1]) > 1000000 ):
                    self.time_vector = 0
                    print("Numerical issues")
                    break
            
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

        l = self.Quad_Length
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

            l = self.Quad_Length

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


      
    
