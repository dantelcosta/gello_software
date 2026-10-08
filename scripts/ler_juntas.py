"""Lê e mostra ao vivo as posições das juntas do GELLO (motores livres, sem torque).

Uso: segura o GELLO na pose que queres tornar a referência de calibração,
lê os valores no ecrã, e carrega Ctrl+C para imprimir a pose final formatada
(pronta a colar em --start-joints).
"""

import sys
import time

import numpy as np

from gello.dynamixel.driver import DynamixelDriver

# Porta do GELLO. Passa outra como 1o argumento se o adaptador USB tiver
# outro nome, ex.: python scripts/ler_juntas.py /dev/ttyUSB0
PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-FTBENXKL"
IDS = [1, 2, 3, 4, 5, 6, 7]  # 6 juntas do braço + gripper

driver = DynamixelDriver(IDS, port=PORT, baudrate=57600)

for _ in range(10):
    driver.get_joints()  # aquecimento

print("A ler... move o GELLO à mão. Ctrl+C para fixar a pose.\n")
try:
    while True:
        j = driver.get_joints()
        arm = "  ".join(f"{x:+.3f}" for x in j[:6])
        print(f"braço (rad): [ {arm} ]   gripper: {np.rad2deg(j[6]):6.1f}deg", end="\r")
        time.sleep(0.15)
except KeyboardInterrupt:
    j = driver.get_joints()
    print("\n\n=== POSE FIXADA ===")
    print("6 juntas (rad) :  " + " ".join(f"{x:.4f}" for x in j[:6]))
    print("6 juntas (graus): " + " ".join(f"{np.rad2deg(x):.1f}" for x in j[:6]))
    print(f"gripper (raw)  :  {j[6]:.4f} rad  ({np.rad2deg(j[6]):.1f} deg)")
