import pickle
import threading
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from dm_control import mjcf

from gello.robots.robot import Robot

assert mujoco.viewer is mujoco.viewer


def attach_hand_to_arm(
    arm_mjcf: mjcf.RootElement,
    hand_mjcf: mjcf.RootElement,
) -> None:
    """Attaches a hand to an arm.

    The arm must have a site named "attachment_site".

    Taken from https://github.com/deepmind/mujoco_menagerie/blob/main/FAQ.md#how-do-i-attach-a-hand-to-an-arm

    Args:
      arm_mjcf: The mjcf.RootElement of the arm.
      hand_mjcf: The mjcf.RootElement of the hand.

    Raises:
      ValueError: If the arm does not have a site named "attachment_site".
    """
    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment site found in the arm model.")

    # Expand the ctrl and qpos keyframes to account for the new hand DoFs.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


def add_table_and_object(arena: mjcf.RootElement) -> None:
    """Adiciona uma mesa solida, uma caixa aberta em cima, e um cubo ao lado.

    Cenario de teste de manipulacao no simulador: o UR5 simulado nao atravessa
    a mesa (colisao real) e o cubo tem freejoint, por isso pode ser agarrado,
    levantado e colocado dentro da caixa.
    """
    # Mesa em frente da base do robo, no eixo Y. Trocar o sinal de table_y
    # para a colocar do outro lado do UR5.
    table_x, table_y = 0.0, -0.5
    table_half_size = 0.4
    table_half_height = 0.05
    table_top_z = 0.10  # altura do tampo

    # Mesa: bloco solido com o tampo a table_top_z
    table = arena.worldbody.add(
        "body", name="table", pos=(table_x, table_y, table_top_z - table_half_height)
    )
    table.add(
        "geom",
        name="table_top",
        type="box",
        size=(table_half_size, table_half_size, table_half_height),
        rgba=(0.55, 0.35, 0.2, 1),
    )

    # Caixa: "tina" aberta em cima, feita com fundo + 4 paredes finas
    wall = 0.005
    inner = 0.12  # meia-largura interior da caixa
    box = arena.worldbody.add("body", name="box", pos=(table_x, table_y, table_top_z))
    box.add(
        "geom", name="box_bottom", type="box",
        size=(inner, inner, wall), pos=(0, 0, wall),
        rgba=(0.2, 0.3, 0.8, 1),
    )
    box.add(
        "geom", name="box_wall_pos_x", type="box",
        size=(wall, inner, 0.05), pos=(inner, 0, 0.05),
        rgba=(0.2, 0.3, 0.8, 1),
    )
    box.add(
        "geom", name="box_wall_neg_x", type="box",
        size=(wall, inner, 0.05), pos=(-inner, 0, 0.05),
        rgba=(0.2, 0.3, 0.8, 1),
    )
    box.add(
        "geom", name="box_wall_pos_y", type="box",
        size=(inner, wall, 0.05), pos=(0, inner, 0.05),
        rgba=(0.2, 0.3, 0.8, 1),
    )
    box.add(
        "geom", name="box_wall_neg_y", type="box",
        size=(inner, wall, 0.05), pos=(0, -inner, 0.05),
        rgba=(0.2, 0.3, 0.8, 1),
    )

    # Cubo ao lado da caixa, em cima da mesa, com freejoint (pode ser
    # agarrado e colocado dentro da caixa).
    cube_half = 0.015  # meio-lado do cubo (3 cm de aresta)
    cube_x = table_x + inner + wall + cube_half + 0.06
    cube_y = table_y
    cube_z = table_top_z + cube_half
    cube = arena.worldbody.add("body", name="cube", pos=(cube_x, cube_y, cube_z))
    cube.add("freejoint", name="cube_joint")
    cube.add(
        "geom", name="cube_geom", type="box",
        size=(cube_half, cube_half, cube_half), rgba=(0.9, 0.2, 0.1, 1),
        mass=0.05,
    )

    # O freejoint do cubo acrescenta 7 valores (posicao+rotacao) ao estado do
    # modelo - o keyframe "home" (pose inicial) tem de ser alargado para
    # incluir esses 7 valores, senao o MuJoCo recusa-se a carregar o modelo.
    keys = arena.find_all("key")
    if len(keys) > 0:
        key = keys[0]
        key.qpos = np.concatenate([key.qpos, [cube_x, cube_y, cube_z, 1, 0, 0, 0]])


def build_scene(robot_xml_path: str, gripper_xml_path: Optional[str] = None):
    # assert robot_xml_path.endswith(".xml")

    arena = mjcf.RootElement()
    arm_simulate = mjcf.from_path(robot_xml_path)
    # arm_copy = mjcf.from_path(xml_path)

    if gripper_xml_path is not None:
        # attach gripper to the robot at "attachment_site"
        gripper_simulate = mjcf.from_path(gripper_xml_path)
        attach_hand_to_arm(arm_simulate, gripper_simulate)

    arena.worldbody.attach(arm_simulate)
    # arena.worldbody.attach(arm_copy)

    add_table_and_object(arena)

    return arena


# Nomes dos dois braços na cena bimanual. Cada braço fica com este prefixo
# nos nomes do MuJoCo (ex. "left/shoulder_pan_joint"). A ordem define a ordem
# do vetor de juntas: primeiro o braço esquerdo (7 valores), depois o direito.
BIMANUAL_ARMS = ("left", "right")

# Os dois braços são o simétrico um do outro (como os dois GELLOs): o
# esquerdo usa a pose de referência espelhada (cotovelo para o outro lado).
# Na pose de referência os dois antebraços apontam para dentro, um para o
# outro, como os GELLOs na bancada. Rodar aqui as bases (graus, em torno do
# eixo vertical) se a montagem real mudar.
BIMANUAL_BASE_YAW_DEG = {"left": 0.0, "right": 0.0}
BIMANUAL_HOME_DEG = {
    "left": [0, -90, -90, -90, 90, 0, 0],
    "right": [0, -90, 90, -90, -90, 0, 0],
}


def add_shared_table_and_objects(arena: mjcf.RootElement, arm_spacing: float) -> None:
    """Mesa comprida em frente dos dois UR5, caixa ao centro e um cubo por braço.

    A caixa fica a meio, ao alcance dos dois braços, para se poder passar um
    cubo de um robô para o outro ou pô-lo dentro da caixa.
    """
    table_y = -0.5
    table_half_x = arm_spacing / 2 + 0.4  # cobre os dois braços
    table_half_y = 0.4
    table_half_height = 0.05
    table_top_z = 0.10

    table = arena.worldbody.add(
        "body", name="table", pos=(0, table_y, table_top_z - table_half_height)
    )
    table.add(
        "geom",
        name="table_top",
        type="box",
        size=(table_half_x, table_half_y, table_half_height),
        rgba=(0.55, 0.35, 0.2, 1),
    )

    # Caixa aberta em cima, ao centro da mesa
    wall = 0.005
    inner = 0.12
    box = arena.worldbody.add("body", name="box", pos=(0, table_y, table_top_z))
    for name, size, pos in (
        ("box_bottom", (inner, inner, wall), (0, 0, wall)),
        ("box_wall_pos_x", (wall, inner, 0.05), (inner, 0, 0.05)),
        ("box_wall_neg_x", (wall, inner, 0.05), (-inner, 0, 0.05)),
        ("box_wall_pos_y", (inner, wall, 0.05), (0, inner, 0.05)),
        ("box_wall_neg_y", (inner, wall, 0.05), (0, -inner, 0.05)),
    ):
        box.add("geom", name=name, type="box", size=size, pos=pos, rgba=(0.2, 0.3, 0.8, 1))

    # Um cubo em frente de cada braço (vermelho à esquerda, verde à direita)
    cube_half = 0.015
    colors = {"left": (0.9, 0.2, 0.1, 1), "right": (0.1, 0.7, 0.2, 1)}
    for side, sign in zip(BIMANUAL_ARMS, (-1, 1)):
        cube = arena.worldbody.add(
            "body",
            name=f"cube_{side}",
            pos=(sign * arm_spacing / 2, table_y, table_top_z + cube_half),
        )
        cube.add("freejoint", name=f"cube_{side}_joint")
        cube.add(
            "geom", name=f"cube_{side}_geom", type="box",
            size=(cube_half, cube_half, cube_half), rgba=colors[side], mass=0.05,
        )


def build_bimanual_scene(
    robot_xml_path: str, gripper_xml_path: str, arm_spacing: float = 1.12
) -> mjcf.RootElement:
    """Dois UR5 com garra, lado a lado no eixo X, virados para a mesma mesa.

    arm_spacing: distância entre as bases dos dois braços, em metros.
    """
    arena = mjcf.RootElement()
    for side, sign in zip(BIMANUAL_ARMS, (-1, 1)):
        arm = mjcf.from_path(robot_xml_path)
        arm.model = side  # prefixo dos nomes, evita nomes repetidos
        attach_hand_to_arm(arm, mjcf.from_path(gripper_xml_path))
        frame = arena.worldbody.attach(arm)
        frame.pos = (sign * arm_spacing / 2, 0, 0)
        half_yaw = np.deg2rad(BIMANUAL_BASE_YAW_DEG[side]) / 2
        frame.quat = (np.cos(half_yaw), 0, 0, np.sin(half_yaw))

    # Cada braço traz o seu keyframe "home" só com as suas juntas; com dois
    # braços e os cubos esses keyframes ficam incompletos e o MuJoCo recusa o
    # modelo. Não são usados pela simulação, por isso saem.
    for key in arena.find_all("key"):
        key.remove()

    add_shared_table_and_objects(arena, arm_spacing)
    return arena


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()


class MujocoRobotServer:
    def __init__(
        self,
        xml_path: str,
        gripper_xml_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
    ):
        self._has_gripper = gripper_xml_path is not None
        arena = build_scene(xml_path, gripper_xml_path)

        assets: Dict[str, str] = {}
        for asset in arena.asset.all_children():
            if asset.tag == "mesh":
                f = asset.file
                assets[f.get_vfs_filename()] = asset.file.contents

        xml_string = arena.to_xml_string()
        # save xml_string to file
        with open("arena.xml", "w") as f:
            f.write(xml_string)

        self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        self._data = mujoco.MjData(self._model)

        self._num_joints = self._model.nu

        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = self._joint_state

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

        self._print_joints = print_joints

    def num_dofs(self) -> int:
        return self._num_joints

    def _read_joint_state(self) -> np.ndarray:
        return self._data.qpos.copy()[: self._num_joints]

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, (
            f"Expected joint state of length {self._num_joints}, "
            f"got {len(joint_state)}."
        )
        if self._has_gripper:
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
        else:
            self._joint_cmd = joint_state.copy()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self._data.qpos.copy()[: self._num_joints]
        joint_velocities = self._data.qvel.copy()[: self._num_joints]
        ee_site = "attachment_site"
        try:
            ee_pos = self._data.site_xpos.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_mat = self._data.site_xmat.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat)
        except Exception:
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
            ee_quat[0] = 1
        gripper_pos = self._data.qpos.copy()[self._num_joints - 1]
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
        }

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                self._data.ctrl[:] = self._joint_cmd
                # self._data.qpos[:] = self._joint_cmd
                mujoco.mj_step(self._model, self._data)
                self._joint_state = self._read_joint_state()

                if self._print_joints:
                    print(self._joint_state)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()


class MujocoBimanualRobotServer(MujocoRobotServer):
    """Dois UR5 com garra na mesma cena, servidos como um único robô de 14 DoF.

    O vetor de juntas é [6 juntas + garra do braço esquerdo, 6 juntas + garra
    do braço direito], o mesmo formato que o BimanualAgent do run_env.py
    (--bimanual) envia.
    """

    def __init__(
        self,
        xml_path: str,
        gripper_xml_path: str,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        arm_spacing: float = 1.12,
    ):
        arena = build_bimanual_scene(xml_path, gripper_xml_path, arm_spacing)

        assets: Dict[str, str] = {}
        for asset in arena.asset.all_children():
            if asset.tag == "mesh":
                f = asset.file
                assets[f.get_vfs_filename()] = asset.file.contents

        self._model = mujoco.MjModel.from_xml_string(arena.to_xml_string(), assets)
        self._data = mujoco.MjData(self._model)
        self._num_joints = self._model.nu

        # Na cena com dois braços o qpos já não começa pelas juntas do robô
        # (cada garra tem 8 juntas internas), por isso as posições são lidas
        # pelo nome de cada junta.
        joint_names = []
        for side in BIMANUAL_ARMS:
            joint_names += [
                f"{side}/{j}"
                for j in (
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                    "robotiq_2f85/right_driver_joint",
                )
            ]
        joint_ids = [self._model.joint(n).id for n in joint_names]
        self._qpos_idx = self._model.jnt_qposadr[joint_ids]
        self._qvel_idx = self._model.jnt_dofadr[joint_ids]
        self._gripper_idx = [6, 13]  # posição das garras no vetor de juntas
        self._ee_site_ids = [
            self._model.site(f"{side}/attachment_site").id for side in BIMANUAL_ARMS
        ]

        # Os dois braços arrancam já na pose de referência (forma em "L").
        # Com todas as juntas a zero cada UR5 fica esticado na horizontal
        # (~1 m) e podia nascer dentro da mesa ou do outro braço.
        home = np.deg2rad(
            np.concatenate([BIMANUAL_HOME_DEG[side] for side in BIMANUAL_ARMS])
        )
        self._data.qpos[self._qpos_idx] = home
        mujoco.mj_forward(self._model, self._data)

        self._joint_state = self._read_joint_state()
        self._joint_cmd = home.copy()

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        self._print_joints = print_joints

    def _read_joint_state(self) -> np.ndarray:
        return self._data.qpos[self._qpos_idx].copy()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, (
            f"Expected joint state of length {self._num_joints}, "
            f"got {len(joint_state)}."
        )
        _joint_state = joint_state.copy()
        # A garra do GELLO vai de 0 (aberta) a 1 (fechada); o atuador da
        # Robotiq 2F-85 no MuJoCo vai de 0 a 255.
        _joint_state[self._gripper_idx] = _joint_state[self._gripper_idx] * 255
        self._joint_cmd = _joint_state

    def get_observations(self) -> Dict[str, np.ndarray]:
        ee_pos_quat = []
        for site_id in self._ee_site_ids:
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, self._data.site_xmat[site_id])
            ee_pos_quat.append(np.concatenate([self._data.site_xpos[site_id], ee_quat]))
        joint_positions = self._read_joint_state()
        return {
            "joint_positions": joint_positions,
            "joint_velocities": self._data.qvel[self._qvel_idx].copy(),
            "ee_pos_quat": np.concatenate(ee_pos_quat),
            # array com as duas garras, para o BimanualAgent dividir ao meio
            "gripper_position": joint_positions[self._gripper_idx],
        }
