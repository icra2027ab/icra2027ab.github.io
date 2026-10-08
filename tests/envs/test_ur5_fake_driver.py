"""The powered-off stand-in must report what the real controller reports."""

import numpy as np

from policy_abstraction.dynamics.ur5_kinematics import ur5e_forward_kinematics
from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS
from policy_abstraction.envs.ur5.fake_driver import FakeUr5Driver

TCP = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]


class TestTcpPose:
    def test_reports_the_tcp_not_the_flange(self):
        """RTDE's getActualTCPPose() includes the controller's TCP offset. The
        runner's forward kinematics applies the same offset to every predicted
        chunk, so a fake that reported the bare flange would put the published
        end-effector 22 cm away from where every predicted trajectory starts."""
        drv = FakeUr5Driver([], 0.02, tcp_offset=TCP)
        pose = np.asarray(drv.rtde_rec.getActualTCPPose())
        pos, rotvec = ur5e_forward_kinematics(UR5E_HOME_JOINT_POSITIONS, TCP)
        np.testing.assert_allclose(pose[:3], pos, atol=1e-9)
        np.testing.assert_allclose(pose[3:], rotvec, atol=1e-9)
        # And it moves with the joints.
        q = np.asarray(UR5E_HOME_JOINT_POSITIONS) + 0.2
        drv.set_joints(q)
        np.testing.assert_allclose(
            np.asarray(drv.rtde_rec.getActualTCPPose())[:3],
            ur5e_forward_kinematics(q, TCP)[0],
            atol=1e-9,
        )

    def test_no_offset_means_the_flange(self):
        drv = FakeUr5Driver([], 0.02)
        pose = np.asarray(drv.rtde_rec.getActualTCPPose())
        pos, _ = ur5e_forward_kinematics(UR5E_HOME_JOINT_POSITIONS)
        np.testing.assert_allclose(pose[:3], pos, atol=1e-9)

    def test_the_control_interface_reports_the_configured_offset(self):
        """What the server's --tcp-offset pre-flight compares against."""
        assert FakeUr5Driver([], 0.02, tcp_offset=TCP).rtde_ctrl.getTCPOffset() == TCP
        assert FakeUr5Driver([], 0.02).rtde_ctrl.getTCPOffset() == [0.0] * 6
