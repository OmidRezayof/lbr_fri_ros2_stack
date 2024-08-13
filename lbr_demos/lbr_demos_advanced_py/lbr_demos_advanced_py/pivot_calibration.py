import numpy as np
import rclpy
import threading
import sys
import select
import lbr_demos_py.asbr 

from geometry_msgs.msg import Pose

from lbr_fri_idl.msg import LBRState

from .admittance_controller import AdmittanceController
from .lbr_base_position_command_node import LBRBasePositionCommandNode

capture_request = False
edge_calib_req = False



def listen_for_keypress(node):
    global capture_request, edge_calib_req
    while rclpy.ok():
        # Check for user input
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line == 'e':
                print("Shutting down...")
                rclpy.shutdown()
                break
            if line == 'c':
                capture_request = True
                print('Pivot Calibration: Capture requested.')
            if line == 'h':
                edge_calib_req = True
                print('Edge Calibration: Capture requested.')

class PivotCalibrationNode(LBRBasePositionCommandNode):
    def __init__(self, node_name: str = "pivot_calib") -> None:
        super().__init__(node_name=node_name)

        # parameters
        self.declare_parameter("base_link", "link_0")
        self.declare_parameter("end_effector_link", "link_ee")
        self.declare_parameter("f_ext_th", [2.0, 2.0, 8.0, 10.0, 10.0, 10.0])
        self.declare_parameter("dq_gains", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("dx_gains", [0.2, 0.2, 0.2, 0.4, 0.4, 0.4])
        self.declare_parameter("exp_smooth", 0.95)

        self._init = False
        self._lbr_state = LBRState()
        self._exp_smooth = (
            self.get_parameter("exp_smooth").get_parameter_value().double_value
        )
        if self._exp_smooth < 0.0 or self._exp_smooth > 1.0:
            raise ValueError("Exponential smoothing factor must be in [0, 1].")

        self._controller = AdmittanceController(
            robot_description=self._robot_description,
            base_link=self.get_parameter("base_link")
            .get_parameter_value()
            .string_value,
            end_effector_link=self.get_parameter("end_effector_link")
            .get_parameter_value()
            .string_value,
            f_ext_th = self.get_parameter('f_ext_th').value,
            dq_gains = self.get_parameter('dq_gains').value,
            dx_gains = self.get_parameter('dx_gains').value,
        )

        # log parameters to terminal
        self._log_parameters()

        self.curr_pose_subs = self.create_subscription(Pose, 'state/pose', self.curr_pose_update, 1)
        self.curr_pose = Pose()

        self.Pivot_Calibration_Poses = []

        self.Edge_Calibration_Poses = []

    def _log_parameters(self) -> None:
        self.get_logger().info("*** Parameters:")
        self.get_logger().info(
            f"*   base_link: {self.get_parameter('base_link').value}"
        )
        self.get_logger().info(
            f"*   end_effector_link: {self.get_parameter('end_effector_link').value}"
        )
        self.get_logger().info(f"*   f_ext_th: {self.get_parameter('f_ext_th').value}")
        self.get_logger().info(f"*   dq_gains: {self.get_parameter('dq_gains').value}")
        self.get_logger().info(f"*   dx_gains: {self.get_parameter('dx_gains').value}")
        self.get_logger().info(f"*   exp_smooth: {self.get_parameter('exp_smooth').value}")

    def _on_lbr_state(self, lbr_state: LBRState) -> None:
        self._smooth_lbr_state(lbr_state)

        lbr_command = self._controller(self._lbr_state, self._dt)
        self._lbr_joint_position_command_pub.publish(lbr_command)

    def _smooth_lbr_state(self, lbr_state: LBRState) -> None:
        if not self._init:
            self._lbr_state = lbr_state
            self._init = True
            return

        self._lbr_state.measured_joint_position = (
            (1 - self._exp_smooth)
            * np.array(self._lbr_state.measured_joint_position.tolist())
            + self._exp_smooth * np.array(lbr_state.measured_joint_position.tolist())
        ).data

        self._lbr_state.external_torque = (
            (1 - self._exp_smooth) * np.array(self._lbr_state.external_torque.tolist())
            + self._exp_smooth * np.array(lbr_state.external_torque.tolist())
        ).data

    def curr_pose_update(self, msg):
        self.curr_pose = msg
        global capture_request, edge_calib_req
        if capture_request: #Capturing the pose of Dali for performing pivot calibration
            self.Pivot_Calibration_Poses.append(msg)
            print('Pivot Calibration: Pose Added! Total number of poses is {}'.format(len(self.Pivot_Calibration_Poses)))
            if(len(self.Pivot_Calibration_Poses)>2):
                tip, divot, resid_dist = self.run_pivot_calibration(self.Pivot_Calibration_Poses)
                print('Tip wrt EE [m]:', tip)
                print('Divot Location [m]:', divot)
                print('Residual distances for each observation [m]:', resid_dist, '\n')
            capture_request = False

        if edge_calib_req: #Capture the edges of a container to find the center of the container
            self.Edge_Calibration_Poses.append(msg)
            print('Edge Calibration: Pose Added! Total number of poses is {}'.format(len(self.Edge_Calibration_Poses)))
            if(len(self.Edge_Calibration_Poses)>2):
                center, radius, height = self.run_edge_calibration(self.Edge_Calibration_Poses)
                print('Center wrt Robot Base Frame: ', center)
                print('Radius: ', radius)
                print('height: ', height)
            edge_calib_req = False






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
