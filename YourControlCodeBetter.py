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
