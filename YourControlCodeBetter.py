import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are

class YourCtrl:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):  # ← MAKE SURE THIS LINE EXISTS
        self.m = m
        self.d = d
        self.init_qpos = d.qpos.copy()

        self.pend_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")
        self.base_act_idx = 0  # assumes base_yaw is first actuator

        self.g = 9.81
        self.l = 0.25
        self.m_pend = 0.2

        A = np.array([[0, 1],
                      [-self.g / self.l, 0]])
        B = np.array([[0],
                      [1 / (self.m_pend * self.l ** 2)]])
        Q = np.diag([150.0, 8.0])
        R = np.array([[0.1]])
        P = solve_continuous_are(A, B, Q, R)
        self.K = (np.linalg.inv(R) @ B.T @ P).flatten()

        self.velocity_damping = 1.0
        self.max_vel = 6.0
        self.max_torque = 100.0

    def CtrlUpdate(self):
        theta = self.d.qpos[self.pend_idx]
        theta_dot = self.d.qvel[self.pend_idx]
        x = np.array([[theta], [theta_dot]])

        # LQR control for pendulum
        u = -self.K @ x
        u = float(np.clip(u, -self.max_torque, self.max_torque))

        # ========================
        # MANUAL JOINT POSITION CLAMPING
        # ========================
        base_joint_idx = self.base_act_idx  # assuming joint index matches actuator index
        base_qpos = self.d.qpos[base_joint_idx]
        base_qvel = self.d.qvel[base_joint_idx]

        # Clamp rotation: if angle exceeds ±pi, reset
        if abs(base_qpos) > np.pi:
            self.d.qpos[base_joint_idx] = np.clip(base_qpos, -np.pi, np.pi)
            self.d.qvel[base_joint_idx] = 0.0  # reset spin

        # Limit base torque application
        self.d.ctrl[base_joint_idx] = u

        # Stabilize all other joints
        for i in range(self.m.nu):
            if i != base_joint_idx:
                qpos_i = self.d.qpos[i]
                qvel_i = self.d.qvel[i]

                # Soft clamp velocity
                if abs(qvel_i) > self.max_vel:
                    self.d.qvel[i] = np.sign(qvel_i) * self.max_vel

                # PD damping
                kp = 200.0
                kd = self.velocity_damping
                torque = kp * (self.init_qpos[i] - qpos_i) - kd * qvel_i
                self.d.ctrl[i] = np.clip(torque, -self.max_torque, self.max_torque)

        return True
    
# import mujoco
# import numpy as np
# from scipy.linalg import inv, eig, solve_continuous_are

# class YourCtrl:
#   def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
#     self.m = m
#     self.d = d
#     self.init_qpos = d.qpos.copy()
#     self.pend_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pend_roll")
#     self.base_act_idx = 0
#     d.qpos[self.pend_idx] = self.init_qpos[self.pend_idx]
#     d.qvel[self.pend_idx] = 0.0
#     self.g = 9.81
#     self.l = 0.25
#     self.m_pend = 0.2
#     A = np.array([[0, 1],
#                   [-self.g / self.l, 0]])
#     B = np.array([[0], 
#                   [1 / (self.m_pend * self.l ** 2)]])
#     Q = np.diag([150.0, 15.0])  # Increase penalty on angular velocity
#     R = np.array([[0.05]])
#     P = solve_continuous_are(A, B, Q, R)
#     self.K = (np.linalg.inv(R) @ B.T @ P).flatten()

#   def CtrlUpdate(self):
#     theta = self.d.qpos[self.pend_idx]
#     theta_dot = self.d.qvel[self.pend_idx]
#     x = np.array([[theta], [theta_dot]])
#     u = -self.K @ x - 0.1 * theta_dot
#     u = float(np.clip(u, -200, 200))
#     self.d.ctrl[self.base_act_idx] = u
#     for i in range(self.m.nu):
#         if i != self.base_act_idx:
#             self.d.ctrl[i] = 200.0 * (self.init_qpos[i] - self.d.qpos[i]) - 15.0 * self.d.qvel[i]
#     return True 
  
# import mujoco
# import numpy as np
# from scipy.linalg import inv, eig

# class YourCtrl:
#   def __init__(self, m:mujoco.MjModel, d: mujoco.MjData):
#     self.m = m
#     self.d = d
#     self.init_qpos = d.qpos.copy()

#     # Control gains (using similar values to CircularMotion)
#     self.kp = 25000.0
#     self.kd = 400.0


# import mujoco
# import numpy as np
# from scipy.linalg import solve_continuous_are

# class YourCtrl:
#     def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
#         self.m = m
#         self.d = d
#         self.pend_joint = mujoco.mj_name2id(
#             m, mujoco.mjtObj.mjOBJ_JOINT, "pendulum"
#         )

#         # === physical params (set to match your XML!) ===
#         self.mass = 0.1
#         self.length = 0.5
#         self.inertia = self.mass * self.length**2
#         self.g = 9.81

#         # Energy-pump gain
#         self.K_e = 2.0

#         # Build LQR around θ=π
#         # State φ = θ−π
#         A = np.array([[0, 1],
#                       [ self.g/self.length, 0]])
#         # *** Corrected sign here: +1/self.inertia ***
#         B = np.array([[0],
#                       [ 1/self.inertia ]])
#         Q = np.diag([10, 1])
#         R = np.array([[0.1]])
#         P = solve_continuous_are(A, B, Q, R)
#         self.K_lqr = np.linalg.inv(R) @ B.T @ P

#         # When |θ−π| smaller than this, switch to LQR
#         self.switch_thresh = 0.2

#     def CtrlUpdate(self) -> bool:
#         θ    = self.d.qpos[self.pend_joint]
#         θdot = self.d.qvel[self.pend_joint]

#         # === total energy relative to downward (θ=0) ===
#         # U = m g l (1 − cosθ),  so U(0)=0, U(π)=2mgl
#         U      = self.mass * self.g * self.length * (1 - np.cos(θ))
#         K      = 0.5 * self.inertia * θdot**2
#         E      = U + K
#         E_des  = 2 * self.mass * self.g * self.length   # <<-- corrected!

#         # swing-up or balance?
#         if abs((θ - np.pi + np.pi) % (2*np.pi) - np.pi) > self.switch_thresh:
#             # swing-up energy pump
#             u = self.K_e * (E_des - E) * np.sign(θdot * np.cos(θ))
#             phase = "SWING"
#         else:
#             # LQR about upright
#             φ = θ - np.pi
#             x = np.array([[φ], [θdot]])
#             u = - (self.K_lqr @ x).item()
#             phase = "LQR"

#         # zero out any other actuators
#         self.d.ctrl[:] = 0.0
#         self.d.ctrl[self.pend_joint] = u

#         # debug print (you can comment these out once it works)
#         print(f"{phase}: θ={θ:.2f}, θdot={θdot:.2f}, E={E:.2f}, E_des={E_des:.2f}, u={u:.2f}")
#         return True
