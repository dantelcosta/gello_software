"""Teleoperacao com retencao resistiva / "mola" no gatilho (experimental).

Substitui, apenas neste processo, o GelloAgent por uma variante que poe os
servos XL330 no modo 5 (Current-based Position Control). Nao altera nada no
repositorio.

Dois usos:
  * retencao resistiva em todas as juntas (alternativa aos elasticos) - testada
    e pouco eficaz, os XL330 nao tem forca suficiente;
  * --gripper-spring: mantem so o servo do gatilho (ID 7) na posicao aberta,
    para servir de "mola" quando a mola fisica do gatilho esta fraca.

Corre no lugar do Terminal 2 (run_env.py). Requer o simulador ou o UR5 no
Terminal 1, como na teleoperacao normal.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict

import numpy as np


# Raiz do repositorio (este ficheiro esta em scripts/).
GELLO_REPO = Path(__file__).resolve().parent.parent

DEFAULT_PORT = "/dev/cu.usbserial-FTBENXKL"
START_JOINTS = (0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0)

CURRENT_BASED_POSITION_MODE = 5
ADDR_TORQUE_ENABLE = 64
ADDR_POSITION_P_GAIN = 84
ADDR_GOAL_PWM = 100
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_PWM = 124
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
LEN_SAFETY_TELEMETRY = 23


if str(GELLO_REPO) not in sys.path:
    sys.path.insert(0, str(GELLO_REPO))

from dynamixel_sdk.group_sync_read import GroupSyncRead  # noqa: E402
from dynamixel_sdk.robotis_def import COMM_SUCCESS  # noqa: E402
from gello.agents import gello_agent as gello_agent_module  # noqa: E402
from gello.dynamixel import driver as dynamixel_driver_module  # noqa: E402
from gello.utils import control_utils  # noqa: E402


OriginalGelloAgent = gello_agent_module.GelloAgent
OriginalDynamixelDriver = dynamixel_driver_module.DynamixelDriver


class ResistiveGelloAgent(OriginalGelloAgent):
    """GelloAgent com referência elasto-plástica e torque sempre limitado."""

    last_instance: "ResistiveGelloAgent | None" = None

    def __init__(
        self,
        *args,
        yield_degrees: float = 2.0,
        motor_currents_ma: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 17),
        gripper_spring: bool = False,
        position_p_gain: int = 55,
        goal_pwm_limit: int = 400,
        max_temperature_c: int = 50,
        hold_hz: float = 15.0,
        telemetry_period_s: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        ResistiveGelloAgent.last_instance = self

        motor_count = self._robot.num_dofs()
        if motor_count != 7:
            raise RuntimeError(
                f"Este teste espera 6 articulações + garra; encontrou {motor_count}"
            )

        self._yield_angles = np.deg2rad(
            [yield_degrees] * (motor_count - 1) + [5.0]
        )
        if len(motor_currents_ma) != motor_count:
            raise RuntimeError("É necessário indicar uma corrente para cada ID 1--7")
        # Um valor zero significa torque realmente desligado nesse motor.
        self._goal_currents = np.asarray(motor_currents_ma, dtype=int)
        self._gripper_spring = bool(gripper_spring)
        self._position_p_gain = int(position_p_gain)
        self._goal_pwm_limit = int(goal_pwm_limit)
        self._max_temperature_c = int(max_temperature_c)
        self._hold_period_s = 1.0 / float(hold_hz)
        self._telemetry_period_s = float(telemetry_period_s)
        self._minimum_command_change = math.radians(0.25)

        self._references: np.ndarray | None = None
        self._last_commanded_references: np.ndarray | None = None
        self._resistive_enabled = False
        self._closed = False
        self._fault: str | None = None
        self._last_act_time = time.monotonic()
        self._last_telemetry_time = 0.0
        self._telemetry_failures = 0
        self._telemetry_group: GroupSyncRead | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._hold_stop = threading.Event()
        self._hold_thread: threading.Thread | None = None

    @property
    def _driver(self):
        return self._robot._driver

    def _check_result(self, result: int, error: int, message: str) -> None:
        if result != COMM_SUCCESS or error != 0:
            raise RuntimeError(f"{message}: comm={result}, dxl_error={error}")

    def _write_initial_configuration(self, positions: np.ndarray) -> None:
        driver = self._driver

        with driver._lock:
            for index, dxl_id in enumerate(driver._ids):
                result, error = driver._packetHandler.write2ByteTxRx(
                    driver._portHandler,
                    dxl_id,
                    ADDR_POSITION_P_GAIN,
                    self._position_p_gain,
                )
                self._check_result(result, error, f"Falha no P gain do ID {dxl_id}")

                result, error = driver._packetHandler.write2ByteTxRx(
                    driver._portHandler,
                    dxl_id,
                    ADDR_GOAL_PWM,
                    self._goal_pwm_limit,
                )
                self._check_result(result, error, f"Falha no PWM do ID {dxl_id}")

                position_raw = int(round(positions[index] * 2048.0 / math.pi))
                result, error = driver._packetHandler.write4ByteTxRx(
                    driver._portHandler,
                    dxl_id,
                    ADDR_GOAL_POSITION,
                    position_raw & 0xFFFFFFFF,
                )
                self._check_result(
                    result, error, f"Falha na posição inicial do ID {dxl_id}"
                )

                result, error = driver._packetHandler.write2ByteTxRx(
                    driver._portHandler,
                    dxl_id,
                    ADDR_GOAL_CURRENT,
                    int(self._goal_currents[index]) & 0xFFFF,
                )
                self._check_result(
                    result, error, f"Falha na corrente inicial do ID {dxl_id}"
                )

    def _disable_free_joint_torque(self) -> None:
        driver = self._driver
        free_ids = tuple(
            dxl_id
            for index, dxl_id in enumerate(driver._ids)
            if self._goal_currents[index] == 0
        )
        missing_ids = [dxl_id for dxl_id in free_ids if dxl_id not in driver._ids]
        if missing_ids:
            raise RuntimeError(f"IDs livres não encontrados: {missing_ids}")

        with driver._lock:
            for dxl_id in free_ids:
                result, error = driver._packetHandler.write1ByteTxRx(
                    driver._portHandler,
                    dxl_id,
                    ADDR_TORQUE_ENABLE,
                    0,
                )
                self._check_result(
                    result,
                    error,
                    f"Falha ao deixar livre o ID {dxl_id}",
                )

    def enable_resistive_hold(self) -> None:
        if self._resistive_enabled:
            return

        driver = self._driver
        if getattr(driver, "_is_fake", False):
            raise RuntimeError("O driver entrou em modo fake; teste físico cancelado")

        print("\nA preparar retenção resistiva com torque reduzido...")
        try:
            driver.set_torque_mode(False)
            driver.set_operating_mode(CURRENT_BASED_POSITION_MODE)
            driver.verify_operating_mode(CURRENT_BASED_POSITION_MODE)

            positions, _ = driver.get_positions_and_velocities()
            self._references = positions.copy()
            self._last_commanded_references = positions.copy()
            self._write_initial_configuration(positions)

            self._telemetry_group = GroupSyncRead(
                driver._portHandler,
                driver._packetHandler,
                ADDR_PRESENT_PWM,
                LEN_SAFETY_TELEMETRY,
            )
            for dxl_id in driver._ids:
                if not self._telemetry_group.addParam(dxl_id):
                    raise RuntimeError(
                        f"Falha ao preparar telemetria do ID {dxl_id}"
                    )

            driver.set_torque_mode(True)
            self._disable_free_joint_torque()
            self._robot._torque_on = True
            self._resistive_enabled = True
            self._last_act_time = time.monotonic()
            self._last_telemetry_time = self._last_act_time

            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="gello-resistive-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

            self._hold_thread = threading.Thread(
                target=self._hold_control_loop,
                name="gello-resistive-hold",
                daemon=True,
            )
            self._hold_thread.start()

            print(
                "RETENÇÃO ATIVA | "
                + " | ".join(
                    f"ID {dxl_id}="
                    + (
                        "OFF"
                        if self._goal_currents[index] == 0
                        else f"{self._goal_currents[index]} mA"
                    )
                    for index, dxl_id in enumerate(driver._ids)
                )
                + " | "
                + (
                    "gatilho=mola | "
                    if self._gripper_spring and self._goal_currents[-1] > 0
                    else "gatilho=retenção normal | "
                )
                + f"cedência={math.degrees(self._yield_angles[0]):.1f}° | "
                f"controlo={1.0 / self._hold_period_s:.0f} Hz"
            )
        except Exception:
            try:
                driver.set_torque_mode(False)
            finally:
                self._robot._torque_on = False
            raise

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(0.05):
            if not self._resistive_enabled:
                continue
            if time.monotonic() - self._last_act_time > 1.5:
                self._trip_fault(
                    "O ciclo principal ficou sem responder durante 1,5 s"
                )
                return

    def _trip_fault(self, message: str) -> None:
        if self._fault is not None:
            return
        self._fault = message
        print(f"\nPARAGEM DE SEGURANÇA: {message}")
        try:
            self._driver.set_torque_mode(False)
        except Exception as exc:
            print(f"Falha ao desligar torque: {exc}")
        self._robot._torque_on = False
        self._resistive_enabled = False

    def _hold_control_loop(self) -> None:
        """Atualiza a retenção sem bloquear os comandos enviados ao UR5."""
        while not self._hold_stop.wait(self._hold_period_s):
            if not self._resistive_enabled:
                continue

            try:
                positions, _ = self._driver.get_positions_and_velocities()
                self._update_references(positions)

                assert self._references is not None
                assert self._last_commanded_references is not None
                command = self._references.copy()
                change = np.max(
                    np.abs(command - self._last_commanded_references)
                )

                # Não repetimos a mesma escrita quando o braço está parado.
                if change >= self._minimum_command_change:
                    with self._driver._lock:
                        self._driver.set_joints(command.tolist())
                    self._last_commanded_references = command

                now = time.monotonic()
                if now - self._last_telemetry_time >= self._telemetry_period_s:
                    self._read_safety_telemetry()
                    self._last_telemetry_time = now
            except Exception as exc:
                self._trip_fault(f"Falha no controlo resistivo: {exc}")
                return

    def _update_references(self, positions: np.ndarray) -> None:
        assert self._references is not None
        displacement = positions - self._references
        yielding = np.abs(displacement) > self._yield_angles
        # No modo mola, a referência do ID 7 fica na posição aberta capturada
        # no arranque. O utilizador vence a corrente ao apertar e, ao largar,
        # o motor regressa a essa referência.
        if self._gripper_spring and self._goal_currents[-1] > 0:
            yielding[-1] = False
        self._references[yielding] = positions[yielding] - np.sign(
            displacement[yielding]
        ) * self._yield_angles[yielding]

    def _read_safety_telemetry(self) -> None:
        driver = self._driver
        group = self._telemetry_group
        if group is None:
            raise RuntimeError("A telemetria de segurança não foi preparada")

        temperatures = []
        voltages = []
        currents = []

        with driver._lock:
            result = None
            for _ in range(3):
                result = group.txRxPacket()
                if result == COMM_SUCCESS:
                    break
                time.sleep(0.01)

            if result != COMM_SUCCESS:
                self._telemetry_failures += 1
                if self._telemetry_failures >= 3:
                    raise RuntimeError(
                        "Telemetria do GELLO falhou 3 vezes consecutivas: "
                        f"comm={result}"
                    )
                print(
                    "\nAviso: resposta de telemetria perdida; "
                    f"nova tentativa no próximo ciclo (comm={result})."
                )
                return

            self._telemetry_failures = 0
            for dxl_id in driver._ids:
                fields = (
                    (ADDR_PRESENT_CURRENT, 2, "corrente"),
                    (ADDR_PRESENT_INPUT_VOLTAGE, 2, "tensão"),
                    (ADDR_PRESENT_TEMPERATURE, 1, "temperatura"),
                )
                for address, length, name in fields:
                    if not group.isAvailable(dxl_id, address, length):
                        raise RuntimeError(
                            f"Telemetria {name} indisponível no ID {dxl_id}"
                        )

                current = group.getData(dxl_id, ADDR_PRESENT_CURRENT, 2)
                if current > 0x7FFF:
                    current -= 0x10000

                voltage = group.getData(
                    dxl_id, ADDR_PRESENT_INPUT_VOLTAGE, 2
                )
                temperature = group.getData(
                    dxl_id, ADDR_PRESENT_TEMPERATURE, 1
                )

                # Os próprios XL330 desligam o torque se uma proteção de
                # hardware configurada for acionada. Aqui monitorizamos as
                # grandezas que permitem interromper o teste antecipadamente.
                if temperature == 0 and voltage == 0:
                    raise RuntimeError(
                        f"Telemetria inválida recebida do motor {dxl_id}"
                    )

                currents.append(current)
                voltages.append(voltage / 10.0)
                temperatures.append(temperature)

        if max(temperatures) >= self._max_temperature_c:
            raise RuntimeError(
                f"Temperatura de segurança atingida: {max(temperatures)} °C"
            )
        if min(voltages) < 3.7 or max(voltages) > 6.0:
            raise RuntimeError(
                f"Tensão fora da faixa: {min(voltages):.1f}--{max(voltages):.1f} V"
            )

        print(
            "\rGELLO | "
            f"temp={max(temperatures):2d} °C | "
            f"tensão={min(voltages):.1f}--{max(voltages):.1f} V | "
            f"corrente_máx={max(abs(value) for value in currents):3d} mA",
            end="",
            flush=True,
        )

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self._fault is not None:
            raise RuntimeError(self._fault)

        if self._resistive_enabled:
            self._last_act_time = time.monotonic()

        return super().act(obs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._watchdog_stop.set()
        self._hold_stop.set()

        if self._hold_thread is not None:
            self._hold_thread.join(timeout=1.0)

        try:
            self._driver.set_torque_mode(False)
            self._robot._torque_on = False
            print("\nTorque do GELLO desligado.")
        except Exception as exc:
            print(f"\nAviso: não foi possível confirmar torque desligado: {exc}")
        finally:
            try:
                self._driver.close()
            except Exception as exc:
                print(f"Aviso ao fechar a porta: {exc}")
            self._resistive_enabled = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument(
        "--motor-currents-ma",
        type=int,
        nargs=7,
        default=(0, 0, 0, 0, 0, 0, 17),
        metavar="mA",
        help="correntes dos IDs 1 2 3 4 5 6 7; zero desliga o torque",
    )
    parser.add_argument(
        "--gripper-spring",
        action="store_true",
        help="mantém o ID 7 na posição aberta capturada no arranque",
    )
    parser.add_argument("--yield-degrees", type=float, default=2.0)
    parser.add_argument("--position-p-gain", type=int, default=55)
    parser.add_argument("--goal-pwm-limit", type=int, default=400)
    parser.add_argument("--max-temperature-c", type=int, default=50)
    parser.add_argument("--hold-hz", type=float, default=15.0)
    parser.add_argument("--telemetry-period-s", type=float, default=5.0)
    args = parser.parse_args()

    if any(not 0 <= value <= 400 for value in args.motor_currents_ma[:6]):
        parser.error("as correntes dos IDs 1--6 devem estar entre 0 e 400 mA")
    if not 0 <= args.motor_currents_ma[6] <= 30:
        parser.error("a corrente do ID 7 deve estar entre 0 e 30 mA")
    if not 0.5 <= args.yield_degrees <= 8.0:
        parser.error("--yield-degrees deve estar entre 0.5 e 8.0")
    if not 5.0 <= args.hold_hz <= 25.0:
        parser.error("--hold-hz deve estar entre 5 e 25")
    if not 2.0 <= args.telemetry_period_s <= 30.0:
        parser.error("--telemetry-period-s deve estar entre 2 e 30")
    return args


def main() -> None:
    cli = parse_args()

    class PhysicalOnlyDynamixelDriver(OriginalDynamixelDriver):
        def __init__(self, *args, **kwargs) -> None:
            # Este teste nunca deve continuar com posições inventadas.
            kwargs["use_fake_fallback"] = False
            super().__init__(*args, **kwargs)

    # DynamixelRobot importa esta classe no momento da construção.
    dynamixel_driver_module.DynamixelDriver = PhysicalOnlyDynamixelDriver

    class ConfiguredResistiveGelloAgent(ResistiveGelloAgent):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(
                *args,
                yield_degrees=cli.yield_degrees,
                motor_currents_ma=tuple(cli.motor_currents_ma),
                gripper_spring=cli.gripper_spring,
                position_p_gain=cli.position_p_gain,
                goal_pwm_limit=cli.goal_pwm_limit,
                max_temperature_c=cli.max_temperature_c,
                hold_hz=cli.hold_hz,
                telemetry_period_s=cli.telemetry_period_s,
                **kwargs,
            )

    # A substituição existe apenas enquanto este processo estiver ativo.
    gello_agent_module.GelloAgent = ConfiguredResistiveGelloAgent

    original_control_loop = control_utils.run_control_loop

    def run_control_loop_with_resistive_hold(env, agent, *args, **kwargs):
        agent.enable_resistive_hold()
        try:
            return original_control_loop(env, agent, *args, **kwargs)
        finally:
            agent.close()

    control_utils.run_control_loop = run_control_loop_with_resistive_hold

    from experiments import run_env

    run_args = run_env.Args(
        agent="gello",
        gello_port=cli.port,
        start_joints=START_JOINTS,
        hz=100,
    )

    try:
        run_env.main(run_args)
    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador.")
    finally:
        instance = ResistiveGelloAgent.last_instance
        if instance is not None:
            instance.close()


if __name__ == "__main__":
    main()
