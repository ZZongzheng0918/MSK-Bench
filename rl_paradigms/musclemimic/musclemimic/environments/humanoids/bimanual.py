import mujoco
from mujoco import MjSpec
from musclemimic_models import get_xml_path

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from musclemimic.environments.humanoids.base_bimanual import BaseBimanualSkeleton


class MyoBimanualArm(BaseBimanualSkeleton):
    """
    Mujoco environment of a bimanual skeleton model with muscle actuation.
    """

    def __init__(
        self,
        disable_fingers: bool = True,
        enable_muscle_length_observations: bool = False,
        enable_muscle_velocity_observations: bool = False,
        enable_muscle_force_observations: bool = False,
        enable_muscle_excitation_observations: bool = False,
        enable_muscle_activation_observations: bool = False,
        **kwargs,
    ) -> None:
        self._disable_fingers = disable_fingers
        self._enable_muscle_length_observations = enable_muscle_length_observations
        self._enable_muscle_velocity_observations = enable_muscle_velocity_observations
        self._enable_muscle_force_observations = enable_muscle_force_observations
        self._enable_muscle_excitation_observations = enable_muscle_excitation_observations
        self._enable_muscle_activation_observations = enable_muscle_activation_observations
        super().__init__(use_muscles=True, **kwargs)

    @property
    def finger_joints(self) -> list[str]:
        return [
            "cmc_flexion_r",
            "cmc_abduction_r",
            "mp_flexion_r",
            "ip_flexion_r",
            "mcp2_flexion_r",
            "mcp2_abduction_r",
            "mcp3_flexion_r",
            "mcp3_abduction_r",
            "mcp4_flexion_r",
            "mcp4_abduction_r",
            "mcp5_flexion_r",
            "mcp5_abduction_r",
            "md2_flexion_r",
            "md3_flexion_r",
            "md4_flexion_r",
            "md5_flexion_r",
            "pm2_flexion_r",
            "pm3_flexion_r",
            "pm4_flexion_r",
            "pm5_flexion_r",
            "cmc_flexion_l",
            "cmc_abduction_l",
            "mp_flexion_l",
            "ip_flexion_l",
            "mcp2_flexion_l",
            "mcp2_abduction_l",
            "mcp3_flexion_l",
            "mcp3_abduction_l",
            "mcp4_flexion_l",
            "mcp4_abduction_l",
            "mcp5_flexion_l",
            "mcp5_abduction_l",
            "md2_flexion_l",
            "md3_flexion_l",
            "md4_flexion_l",
            "md5_flexion_l",
            "pm2_flexion_l",
            "pm3_flexion_l",
            "pm4_flexion_l",
            "pm5_flexion_l",
        ]

    @property
    def finger_muscles(self) -> list[str]:
        return [
            "FDS2",
            "FDS3",
            "FDS4",
            "FDS5",
            "FDP2",
            "FDP3",
            "FDP4",
            "FDP5",
            "EDC2",
            "EDC3",
            "EDC4",
            "EDC5",
            "EDM",
            "EIP",
            "EPL",
            "EPB",
            "FPL",
            "APL",
            "OP",
            "RI2",
            "RI3",
            "RI4",
            "RI5",
            "LU_RB2",
            "LU_RB3",
            "LU_RB4",
            "LU_RB5",
            "UI_UB2",
            "UI_UB3",
            "UI_UB4",
            "UI_UB5",
            "FDS2_left",
            "FDS3_left",
            "FDS4_left",
            "FDS5_left",
            "FDP2_left",
            "FDP3_left",
            "FDP4_left",
            "FDP5_left",
            "EDC2_left",
            "EDC3_left",
            "EDC4_left",
            "EDC5_left",
            "EDM_left",
            "EIP_left",
            "EPL_left",
            "EPB_left",
            "FPL_left",
            "APL_left",
            "OP_left",
            "RI2_left",
            "RI3_left",
            "RI4_left",
            "RI5_left",
            "LU_RB2_left",
            "LU_RB3_left",
            "LU_RB4_left",
            "LU_RB5_left",
            "UI_UB2_left",
            "UI_UB3_left",
            "UI_UB4_left",
            "UI_UB5_left",
        ]

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        if self._disable_fingers:
            joints_to_remove = []
            for joint in spec.joints:
                if any(finger_joint in joint.name for finger_joint in self.finger_joints):
                    joints_to_remove.append(joint)

            for joint in joints_to_remove:
                spec.delete(joint)

            actuators_to_remove = []
            for actuator in spec.actuators:
                if any(finger_muscle in actuator.name for finger_muscle in self.finger_muscles):
                    actuators_to_remove.append(actuator)

            for actuator in actuators_to_remove:
                spec.delete(actuator)

            tendons_to_remove = []
            for tendon in spec.tendons:
                if any(finger_muscle in tendon.name for finger_muscle in self.finger_muscles):
                    tendons_to_remove.append(tendon)

            for tendon in tendons_to_remove:
                spec.delete(tendon)

        for body_name, site_name in self.body2sites_for_mimic.items():
            body = spec.body(body_name)
            body.add_site(
                name=site_name,
                group=4,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.075, 0.05, 0.025],
                rgba=[1.0, 0.0, 0.0, 0.5],
                pos=[0.0, 0.0, 0.0],
            )

        return spec

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        j_names = [joint.name for joint in spec.joints]

        obs_spec = []

        if self._enable_joint_pos_observations:
            obs_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))

        if self._enable_joint_vel_observations:
            obs_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        for actuator in spec.actuators:
            actuator_name = actuator.name

            if self._enable_muscle_length_observations:
                obs_spec.append(
                    ObservationType.ActuatorLength(
                        f"muscle_length_{actuator_name.lower()}",
                        xml_name=actuator_name,
                    )
                )
            if self._enable_muscle_velocity_observations:
                obs_spec.append(
                    ObservationType.ActuatorVelocity(
                        f"muscle_velocity_{actuator_name.lower()}",
                        xml_name=actuator_name,
                    )
                )
            if self._enable_muscle_force_observations:
                obs_spec.append(
                    ObservationType.ActuatorForce(
                        f"muscle_force_{actuator_name.lower()}",
                        xml_name=actuator_name,
                    )
                )
            if self._enable_muscle_excitation_observations:
                obs_spec.append(
                    ObservationType.ActuatorExcitation(
                        f"muscle_excitation_{actuator_name.lower()}",
                        xml_name=actuator_name,
                    )
                )
            if self._enable_muscle_activation_observations:
                obs_spec.append(
                    ObservationType.ActuatorActivation(
                        f"muscle_activation_{actuator_name.lower()}",
                        xml_name=actuator_name,
                    )
                )

        return obs_spec

    def _get_action_specification(self, spec: MjSpec) -> list[str]:
        return [actuator.name for actuator in spec.actuators]

    @info_property
    def body2sites_for_mimic(self) -> dict[str, str]:
        return {
            "thorax": "upper_body_mimic",
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
        }

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        return get_xml_path("bimanual").as_posix()


class MjxMyoBimanualArm(MyoBimanualArm):
    """
    Mjx version of MyoBimanualArm.
    """

    mjx_enabled = True

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        mjx_backend: str = "jax",
        **kwargs,
    ):
        goal_related_params = [
            "visualize_goal",
            "enable_enhanced_visualization",
            "target_geom_rgba",
            "n_step_lookahead",
            "goal_type",
            "goal_params",
        ]

        extracted_goal_params = {}
        for param in goal_related_params:
            if param in kwargs:
                extracted_goal_params[param] = kwargs.pop(param)

        if "model_option_conf" not in kwargs:
            model_option_conf = {
                "iterations": 4,
                "ls_iterations": 8,
                "disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
            }
        else:
            model_option_conf = kwargs.pop("model_option_conf")

        kwargs.update(extracted_goal_params)

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            mjx_backend=mjx_backend,
            **kwargs,
        )
