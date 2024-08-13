import numpy as np
import rclpy
import threading
import sys
import select
import lbr_demos_py.asbr 
from rclpy.node import Node

from geometry_msgs.msg import Pose

capture_request = 'None'

def listen_for_keypress(node):
    global capture_request
    while rclpy.ok():
        # Check for user input
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line == 'q':
                print("Shutting down...")
                rclpy.shutdown()
                break
            if line == 'p':
                capture_request = line
                print('Pivot Calibration: Capture requested.')
            if line == 'h':
                capture_request = line
                print('Height Calibration: Capture requested.')
            if line == 'e':
                capture_request = line
                print('Edge Calibration: Capture requested.')


class PivotCalibrationNode(Node):
    def __init__(self, node_name: str = "pivot_calib") -> None:
        super().__init__(node_name=node_name)

        # parameters
        self._init = False

        self.curr_pose_subs = self.create_subscription(Pose, 'state/pose', self.curr_pose_update, 1)
        self.curr_pose = Pose()

        self.Pivot_Calibration_Poses = []
        self.Height_Calibration_Poses = []
        self.Edge_Calibration_Poses = []

    
    def curr_pose_update(self, msg):
        self.curr_pose = msg
        global capture_request
        if not self._init:
            print('Enter "p" for capturing a Pivot Calibration pose,')
            print('Enter "e" for capturing an Edge Calibration pose,')
            print('Enter "h" for capturing a Height Calibration pose,')
            print('Enter "q" to quit.')
            self._init = True
            return

        # NOTE: All the positions (except for pivot calibration) are reported as the position of the EE after the calibraiton.
         
        if capture_request == 'p': #Capturing the pose of Dali for performing pivot calibration
            self.Pivot_Calibration_Poses.append(msg)
            print('Pivot Calibration: Pose Added! Total number of poses is {}'.format(len(self.Pivot_Calibration_Poses)))
            if(len(self.Pivot_Calibration_Poses)>2):
                tip, divot, resid_dist = self.run_pivot_calibration(self.Pivot_Calibration_Poses)
                print('Tip wrt EE [m]:', tip)
                print('Divot Location [m]:', divot)
                print('Residual distances for each observation [m]:', resid_dist, '\n')
            capture_request = 'None'

        if capture_request == 'e': #Capture the edges of a container to find the center of the container
            self.Edge_Calibration_Poses.append(msg)
            print('Edge Calibration: Pose Added! Total number of poses is {}'.format(len(self.Edge_Calibration_Poses)))
            if(len(self.Edge_Calibration_Poses)>2):
                center, radius, height = self.run_edge_calibration(self.Edge_Calibration_Poses)
                print('Center wrt Robot Base Frame: ', center)
                print('Radius: ', radius)
                print('height: ', height)
            capture_request = 'None'

        if capture_request == 'h': #Capture the height of the material inside the container to find the height (level) of the material
            self.Height_Calibration_Poses.append(msg)
            print('Height Calibration: Pose Added! Total number of poses is {}'.format(len(self.Height_Calibration_Poses)))

            height = np.array([point.position.z for point in self.Height_Calibration_Poses])
            height = np.mean(height)
            print('Average Material Height: ', height)
            
            capture_request = 'None'



    def run_pivot_calibration(self, poses):
        # poses: a list of geometry_msgs.msg.Pose objects, containing position (x,y,z)
        #        and orientation (x,y,z,w) as quat
        n = len(poses)
        A = np.empty((n * 3, 6))
        b = np.empty((n * 3, 1))
        for pi,p in enumerate(poses):
            rbase = 3 * pi
            R = lbr_demos_py.asbr.Rotation(p.orientation)
            b[rbase + 0, 0] = -p.position.x
            b[rbase + 1, 0] = -p.position.y
            b[rbase + 2, 0] = -p.position.z
            A[rbase:rbase+3, 0:3] = R.m
            A[rbase:rbase+3, 3:6] = -np.eye(3)

        x = np.linalg.lstsq(A, b, rcond=None)[0]

        resid = np.matmul(A, x) - b
        resid_dist = np.sqrt(np.sum(np.square(resid.reshape((3,-1))), axis=0))
        tip = (x[0,0], x[1,0], x[2,0])
        divot = (x[3,0], x[4,0], x[5,0])
        return (tip, divot, resid_dist)

    def run_edge_calibration(self, points):

        # Convert the points into numpy arrays for easier manipulation
        x = np.array([point.position.x for point in points])
        y = np.array([point.position.y for point in points])
        z = np.array([point.position.z for point in points])
        center, radius = self.fit_circle_to_points(x,y)

        z = np.mean(z)

        return center, radius, z



    def fit_circle_to_points(self, x, y):  #Fits the best circle to the points (x,y)
        # Define the A matrix (for the least squares fitting)
        A = np.c_[x*2, y*2, np.ones(len(x))]
        
        # Define the B matrix (distances from origin squared)
        B = x**2 + y**2
        
        # Solve the least squares problem
        C, D, E = np.linalg.lstsq(A, B, rcond=None)[0]
        
        # The center of the circle
        center_x = C
        center_y = D
        
        # Radius of the circle
        radius = np.sqrt(E + center_x**2 + center_y**2)
        
        return (center_x, center_y), radius



def main(args=None) -> None:
    rclpy.init(args=args)
    admittance_control_node = PivotCalibrationNode()

    # Start the keypress listener thread
    keypress_thread = threading.Thread(target=listen_for_keypress, args=(admittance_control_node,), daemon=True)
    keypress_thread.start()

    try:
        rclpy.spin(admittance_control_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
