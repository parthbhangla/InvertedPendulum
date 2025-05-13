# Library imports
import mujoco
import numpy as np
from scipy.linalg import inv, eig, solve_continuous_are

class YourCtrl:
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    # Model and data
    self.m = m
    self.d = d

    # Copying the initial state of the model - for future reference
    self.init_qpos = d.qpos.copy()

    # Getting the index of the pendulum joint named "pend_roll"
    self.pend_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")

    # Index of the base actuator
    self.base_act_idx = 0

    # Pendulum Initialization
    d.qpos[self.pend_idx] = 0.0
    d.qvel[self.pend_idx] = 0.0

    # Constants: gravity, pendulum length, and mass
    self.g = 9.81
    self.l = 0.25
    self.m_pend = 0.2

    # Define the linearized state-space model for the pendulum
    # x = [theta, theta_dot], where theta is the angle
    A = np.array([[0, 1],
                  [-self.g / self.l, 0]])  # Dynamics matrix

    B = np.array([[0], 
                  [1 / (self.m_pend * self.l ** 2)]])  # Control input matrix

    # Define the cost matrices for Linear Quadratic Regulator
    # Q penalizes the state error - the difference between the desired and actual state
    # R penalizes the control effort - the amount of control input applied
    Q = np.diag([150.0, 5.0])
    R = np.array([[0.05]])

    # Solve the continuous-time algebraic Riccati equation (CARE) - to find the solution P
    # Used to calculate the optimal feedback gain matrix K
    P = solve_continuous_are(A, B, Q, R)

    # Compute the optimal LQR gain matrix K - feedback gain - to minimize the cost function
    self.K = (np.linalg.inv(R) @ B.T @ P).flatten()

  def CtrlUpdate(self):
    # Read the pendulum angle and angular velocity
    theta = self.d.qpos[self.pend_idx]
    theta_dot = self.d.qvel[self.pend_idx]

    # Form the state vector
    x = np.array([[theta], [theta_dot]])

    # Compute the control input using LQR: u = -Kx
    u = -self.K @ x

    # Max torque limit (-200 to 200)
    u = float(np.clip(u, -200, 200))

    # Apply control to the base actuator
    self.d.ctrl[self.base_act_idx] = u

    # Apply PD control to stabilize all other joints to initial positions
    for i in range(self.m.nu):  # Loop over all actuators except the base actuator
        if i != self.base_act_idx:
            # PD control: torque = Kp * error - Kd * velocity
            self.d.ctrl[i] = 300.0 * (self.init_qpos[i] - self.d.qpos[i]) - 15.0 * self.d.qvel[i]

    return True
