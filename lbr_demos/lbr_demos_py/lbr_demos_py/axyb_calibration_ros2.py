#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi, sin, cos, asin, atan2, sqrt
from time import sleep

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool, Float32
from lbr_fri_idl.srv import MoveToPose
import numpy as np



from geometry_msgs import msg as geomsg

import lbr_demos_py.asbr as asbr
import tf2_ros
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import os

class axyb_calibration_node(Node):

    def __init__(self):
        super().__init__('axyb_calibration_ros2')
        self.client = self.create_client(MoveToPose, 'move_to_pose')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = MoveToPose.Request()
        self.subs = self.create_subscription(Bool, 'goal_reached_top', self.goal_reach_update, 1)
        self.curr_pose_subs = self.create_subscription(Pose, 'state/pose', self.curr_pose_update, 1)

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer, self)


        self.curr_pose = Pose()

        self.goal_state = False
        self.commiunication_rate = 0.01

        self.is_init = False

    def send_request(self, goal_pose, lin_vel = 0.005):
        self.request.goal_pose = goal_pose
        self.request.lin_vel = Float32(data = lin_vel)
        print('lin_vel set to ', lin_vel)
        self.future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def goal_reach_update(self, msg):
        if msg.data == True:
            self.goal_state = True

    def wait_for_goal(self):
        while not self.goal_state:
            rclpy.spin_once(self, timeout_sec = self.commiunication_rate)
        self.goal_state = False
        return 

    def curr_pose_update(self, msg):
        self.curr_pose = msg
        self.is_init = True

    def wait_for_init(self):
        while not self.is_init:
            rclpy.spin_once(self, timeout_sec = self.commiunication_rate)
        return 

    def optical_pose_callback(self):
        try:
            # Lookup the latest transform by providing the current time or Time(0) for the latest available
            t12 = self.tfBuffer.lookup_transform('optical_tracker', 'tool1', rclpy.time.Time())
            t13 = self.tfBuffer.lookup_transform('optical_tracker', 'tool2', rclpy.time.Time())
            t14 = self.tfBuffer.lookup_transform('optical_tracker', 'tool3', rclpy.time.Time())

            # Access the transform data
            t12 = t12.transform
            t13 = t13.transform
            t14 = t14.transform

            return t12, t13, t14

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'Could not transform: {e}')
            return None, None, None
    
    def go_home(self, lin_vel = 0.005):
        home_pose = Pose()
        home_pose.position.x = 0.4
        home_pose.position.y = 0.0
        home_pose.position.z = 0.655
        home_pose.orientation = (asbr.Rotation.from_ABC([180,0,180],True)).as_geometry_orientation()
        response = self.send_request(home_pose, lin_vel)
        print(response)
        self.wait_for_goal()    
        return 

    def is_close(self, new_pose, pos_thresh = 0.15):
        translation_vec = np.asarray([self.curr_pose.position.x - new_pose.position.x,
                                      self.curr_pose.position.y - new_pose.position.y,
                                      self.curr_pose.position.z - new_pose.position.z])
        if np.linalg.norm(translation_vec) < pos_thresh:
            print('Poses too close. Finding another random pose.')
        return np.linalg.norm(translation_vec) < pos_thresh




"""
    + Simultaneous Robot/World and Tool/Flange Calibration:    
    Implementation of Shah, Mili. "Solving the robot-world/hand-eye calibration problem using the Kronecker product." 
    Journal of Mechanisms and Robotics 5.3 (2013): 031007.
    
    Batch_Processing solvesfor  X and Y in AX=YB from a set of (A,B) paired measurements.
    (Ai,Bi) are absolute pose measurements with known correspondance       
    A: (4x4xn) 
    X: (4x4): unknown
    Y: (4x4): unknown
    B: (4x4xn) 
    n number of measurements
    
    
    @author: elif.ayvali
"""



class axzb_calibration:
    def run_calibration(poses,tfs):
        # Runs AX=ZB Calibration similar to asbr. Reference on calibrateRobotWorldHandEye() at https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga41b1a8dd70eae371eba707d101729c36
        # A = Polaris wrt Tracker
        # B = World wrt KUKA EE
        # X = Polaris wrt World
        # Z = KUKA EE wrt Tracker


        A_rot = []
        A_tran = []
        B_rot = []
        B_tran = []

        for pi,(p,t) in enumerate(zip(poses,tfs)): #umm, pretty sure all the labels are backwards on this. tfs should've been polaris
            rbase = 3 * pi
            # Poses are Tracker wrt Polaris
            #R = asbr.Rotation(p.orientation)
            tracker_wrt_polaris = asbr.Transformation(asbr.Translation(p.position),asbr.Rotation(p.orientation))
            polaris_wrt_tracker = asbr.Transformation(np.linalg.inv(tracker_wrt_polaris.m))


            kuka_wrt_s = asbr.Transformation(asbr.Translation(t.translation),asbr.Rotation(t.rotation))
            s_wrt_kuka = asbr.Transformation(np.linalg.inv(kuka_wrt_s.m))


            A_rot.append(polaris_wrt_tracker.rotation.m)
            A_tran.append(polaris_wrt_tracker.translation.m)
            B_rot.append(s_wrt_kuka.rotation.m)
            B_tran.append(s_wrt_kuka.translation.m)
            

        

        #calib_method = cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        calib_method = cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI
        X_rot, X_tran, Z_rot, Z_tran = cv2.calibrateRobotWorldHandEye(A_rot,A_tran,B_rot,B_tran,calib_method)
        X = asbr.Transformation(asbr.Translation(X_tran),asbr.Rotation(X_rot))
        Z = asbr.Transformation(asbr.Translation(Z_tran),asbr.Rotation(Z_rot))


        return X, Z
    
    def error_calculation(poses,tfs,polaris_wrt_s,kuka_wrt_tracker):
        #Calculates the error in the AXZB Calculation
        error=[]

        for pi,(p,t) in enumerate(zip(poses,tfs)):
            rbase = 3 * pi
            tracker_wrt_polaris = asbr.Transformation(asbr.Translation(p.position),asbr.Rotation(p.orientation))
            polaris_wrt_tracker = asbr.Transformation(np.linalg.inv(tracker_wrt_polaris.m))
            kuka_wrt_s = asbr.Transformation(asbr.Translation(t.translation),asbr.Rotation(t.rotation))
            s_wrt_kuka = asbr.Transformation(np.linalg.inv(kuka_wrt_s.m))
            s_wrt_polaris = asbr.Transformation(np.linalg.inv(polaris_wrt_s.m))
            tracker_wrt_kuka = asbr.Transformation(np.linalg.inv(kuka_wrt_tracker.m))

            kuka_wrt_polaris = asbr.Transformation(np.matmul(kuka_wrt_s.m,s_wrt_polaris.m))
            tracker_wrt_polaris2 = asbr.Transformation(np.matmul(tracker_wrt_kuka.m,kuka_wrt_polaris.m))
            error_point = euclidean_distance(tracker_wrt_polaris.translation.x, tracker_wrt_polaris.translation.y, tracker_wrt_polaris.translation.z, tracker_wrt_polaris2.translation.x, tracker_wrt_polaris2.translation.y, tracker_wrt_polaris2.translation.z)
            error.append(error_point)
            mean_error = sum(error) / len(error)

        return mean_error
        
class axxb_calibration:
    def run_calibration1(poses,tfs):
        # Runs AX=XB Calibration similar to asbr. Reference on calibrateHandEye() at https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga41b1a8dd70eae371eba707d101729c36
        # A = World wrt Kuka EE
        # B = Tracker wrt Polaris
        # X = Polaris wrt World

        A_rot = []
        A_tran = []
        B_rot = []
        B_tran = []

        for pi,(p,t) in enumerate(zip(poses,tfs)):
            rbase = 3 * pi

            tracker_wrt_polaris = asbr.Transformation(asbr.Translation(t.translation),asbr.Rotation(t.rotation))


            kuka_wrt_s = asbr.Transformation(asbr.Translation(p.position),asbr.Rotation(p.orientation))
            s_wrt_kuka = asbr.Transformation(np.linalg.inv(kuka_wrt_s.m))


            A_rot.append(s_wrt_kuka.rotation.m)
            A_tran.append(s_wrt_kuka.translation.m)
            B_rot.append(tracker_wrt_polaris.rotation.m)
            B_tran.append(tracker_wrt_polaris.translation.m)
            

        

        calib_method = cv2.CALIB_HAND_EYE_DANIILIDIS 
        X_rot, X_tran = cv2.calibrateHandEye(A_rot,A_tran,B_rot,B_tran,calib_method)
        X = asbr.Transformation(asbr.Translation(X_tran),asbr.Rotation(X_rot))

        return X
    
    def run_calibration2(poses,tfs):
        # Runs AX=XB Calibration similar to asbr. Reference on calibrateHandEye() at https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga41b1a8dd70eae371eba707d101729c36
        # A = Kuka ee wrt World
        # B = Polaris wrt Tracker
        # X = Kuka ee wrt Tracker

        A_rot = []
        A_tran = []
        B_rot = []
        B_tran = []

        for pi,(p,t) in enumerate(zip(poses,tfs)):
            rbase = 3 * pi

            tracker_wrt_polaris = asbr.Transformation(asbr.Translation(t.translation),asbr.Rotation(t.rotation))
            polaris_wrt_tracker = asbr.Transformation(np.linalg.inv(tracker_wrt_polaris.m))


            kuka_wrt_s = asbr.Transformation(asbr.Translation(p.position),asbr.Rotation(p.orientation))


            A_rot.append(kuka_wrt_s.rotation.m)
            A_tran.append(kuka_wrt_s.translation.m)
            B_rot.append(polaris_wrt_tracker.rotation.m)
            B_tran.append(polaris_wrt_tracker.translation.m)
            

        calib_method = cv2.CALIB_HAND_EYE_HORAUD
        X_rot, X_tran = cv2.calibrateHandEye(A_rot,A_tran,B_rot,B_tran,calib_method)
        X = asbr.Transformation(asbr.Translation(X_tran),asbr.Rotation(X_rot))

        return X

def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def error_filter(list1, list2, list3, max_error):
    """
    Remove values greater than the max error from corresponding positions in three lists.

    Parameters:
    - list1 (list): The first input list.
    - list2 (list): The second input list.
    - list3 (list): The third input list.

    Returns:
    - tuple: A tuple containing modified versions of list1, list2, and list3.
    """
    filtered_lists = zip(list1, list2, list3)
    filtered_lists = [(a, b, c) for a, b, c in filtered_lists if a <= max_error]

    # Unzip the filtered lists to get three separate lists
    filtered_list1, filtered_list2, filtered_list3 = zip(*filtered_lists)

    return filtered_list1, filtered_list2, filtered_list3

def error_calculation(poses,tfs,polaris_wrt_s,kuka_wrt_tracker,str):
    x1_plot=[]
    y1_plot=[]
    z1_plot=[]
    x2_plot=[]
    y2_plot=[]
    z2_plot=[]

    for i,(p,t) in enumerate(zip(poses,tfs)):
        tracker_wrt_polaris = asbr.Transformation(asbr.Translation(p.position),asbr.Rotation(p.orientation))
        polaris_wrt_tracker=asbr.Transformation(np.linalg.inv(tracker_wrt_polaris.m))
        kuka_wrt_s = asbr.Transformation(asbr.Translation(t.translation),asbr.Rotation(t.rotation))
        #polaris_wrt_s
        tracker_wrt_kuka=asbr.Transformation(np.linalg.inv(kuka_wrt_tracker.m))

        polaris_wrt_kuka=asbr.Transformation(np.matmul(tracker_wrt_kuka.m,polaris_wrt_tracker.m))
        polaris_wrt_s_temp=asbr.Transformation(np.matmul(kuka_wrt_s.m,polaris_wrt_kuka.m))
        polaris_wrt_s_2=asbr.Transformation(np.linalg.inv(polaris_wrt_s_temp.m))
        x1_plot.append(polaris_wrt_s.translation.x*1000)
        y1_plot.append(polaris_wrt_s.translation.y*1000)
        z1_plot.append(polaris_wrt_s.translation.z*1000)
        x2_plot.append(polaris_wrt_s_2.translation.x*1000)
        y2_plot.append(polaris_wrt_s_2.translation.y*1000)
        z2_plot.append(polaris_wrt_s_2.translation.z*1000)



    error=[euclidean_distance(x1, y1, z1, x2, y2, z2)
                for x1, y1, z1, x2, y2, z2 in zip(x1_plot, y1_plot, z1_plot, x2_plot, y2_plot, z2_plot)]
    mean_error = sum(error) / len(error)
    print('Avg Error: ',mean_error)
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(x2_plot,y2_plot,z2_plot,color='red',marker='o',s=5,zorder=1)
    ax.scatter(x1_plot,y1_plot,z1_plot,color='blue',marker='+',s=40,zorder=0)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    #add length of poses to title
    ax.set_title(f'{str} Calibration Points ({len(poses)} Poses)')
    #add legend
    ax.legend(['Pose Positions','Calibrated'])
    plt.show()

    return error

def main():
    #Init

    rclpy.init(args=None)
    client = axyb_calibration_node()
    client.wait_for_init()


    folder_path = '/home/omid/Sheela/ct-sdr/src/iiwa_stack_test/axyb_calibration_data/'


    KUKA_pose=Pose()

    t12=[]
    t13=[]
    t14=[]


    poses = []
    tfs = []
    tfs_3=[]
    tfs_4=[]

     #Gathering poses

    test_num = 1
    pose_num = 20

    poses = []
    tfs = []
    tfs_3=[]
    tfs_4=[]

    #poser.move_sync_cart_pos(*home_pose, timeout_ms=15000)


    #np.random.seed(6) #All of these seeds are confirmed have 200 poses without workspace errors
    #np.random.seed(36)
    #np.random.seed(66)

    for i in range(pose_num):
        #Move to place

        place = np.random.uniform([0.550, -0.225, 0.380], [0.600, 0.225, 0.560]) #INCREASE ROTATIONS
        reach_pose = geomsg.Pose()
        reach_pose.position.x = place[0]
        reach_pose.position.y = place[1]
        reach_pose.position.z = place[2]
        reach_pose.orientation = (asbr.Rotation.from_ABC([180.0,0.0,180.0],True)).as_geometry_orientation() 
        lin_vel = 0.02 #m/s TODO: check if the position is very close to where robot is rn

        while(client.is_close(reach_pose)):
            place = np.random.uniform([0.550, -0.225, 0.380], [0.600, 0.225, 0.560]) #INCREASE ROTATIONS
            reach_pose = geomsg.Pose()
            reach_pose.position.x = place[0]
            reach_pose.position.y = place[1]
            reach_pose.position.z = place[2]
            reach_pose.orientation = (asbr.Rotation.from_ABC([180.0,0.0,180.0],True)).as_geometry_orientation()


        response = client.send_request(reach_pose, lin_vel)
        print(f'Success: {response.success}')
        client.wait_for_goal()
        print('at pose')
        sleep(1.0)

        #Update poses,tfs
        poses.append(client.curr_pose)
        t12,t13,t14 = client.optical_pose_callback()
        tfs.append(t12)
        tfs_3.append(t13)
        tfs_4.append(t14)
        print('Pose added! Total number of poses: {',format(len(poses)),'}')
        sleep(0.25)
        
        # make a new directory for pickle files
        poses_path = folder_path + f'poses_{test_num}.pkl'
        tfs_path = folder_path + f'tfs_{test_num}.pkl'
        tfs_3_path = folder_path + f'tfs_3_{test_num}.pkl'
        tfs_4_path = folder_path + f'tfs_4_{test_num}.pkl'
        # make directory if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(poses_path, 'wb') as f:
            pickle.dump(poses, f)
        with open(tfs_path, 'wb') as f:
            pickle.dump(tfs, f)
        with open(tfs_3_path, 'wb') as f:
            pickle.dump(tfs_3, f)
        with open(tfs_4_path, 'wb') as f:
            pickle.dump(tfs_4, f)
        #Repeat  

    client.go_home(0.01)
    # dump poses and tfs to pickle

if __name__ == '__main__':
    main()


