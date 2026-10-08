"""Configura UM servo Dynamixel de cada vez: le o ID atual e poe o novo.

Regra de ouro: liga SO UM servo ao U2D2 quando corres isto. Se tiveres varios
ligados, todos com ID de fabrica (1), vao colidir e nada funciona.

Uso:
    # so ver o que esta ligado (nao mexe em nada):
    venv/bin/python3 scripts/set_servo_id.py

    # por o servo ligado com o ID 3 e baudrate 57600:
    venv/bin/python3 scripts/set_servo_id.py 3

    # idem, noutra porta:
    venv/bin/python3 scripts/set_servo_id.py 3 /dev/cu.usbserial-FTBENXKL

Ordem do GELLO: 1 = base ... 6 = ultimo do braco, 7 = garra.
"""

import sys

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

# --- Enderecos (protocolo 2.0, X-series: XL330/XC330/XL430/XM) ---
ADDR_ID = 7
ADDR_BAUD = 8
ADDR_TORQUE_ENABLE = 64
TARGET_BAUD = 57600
BAUD_CODE_57600 = 1  # valor a escrever em ADDR_BAUD para 57600
SCAN_BAUDS = [57600, 1000000, 2000000, 115200, 3000000, 4000000, 9600]

target_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
port_name = sys.argv[2] if len(sys.argv) > 2 else "/dev/cu.usbserial-FTBENXKL"

if target_id is not None and not (0 <= target_id <= 252):
    sys.exit("ID de destino tem de estar entre 0 e 252")

ph = PortHandler(port_name)
pk = PacketHandler(2.0)
if not ph.openPort():
    sys.exit(f"Falhou abrir a porta {port_name}")

# 1) Encontrar o servo: varre baudrates, broadcast ping
found = None  # (baud, id, model)
for baud in SCAN_BAUDS:
    if not ph.setBaudRate(baud):
        continue
    data, comm = pk.broadcastPing(ph)
    if comm == COMM_SUCCESS and data:
        ids = list(data.keys())
        if len(ids) > 1:
            ph.closePort()
            sys.exit(
                f"@ {baud} baud vejo VARIOS servos {ids}. Liga so UM de cada vez."
            )
        sid = ids[0]
        model = data[sid][0]
        found = (baud, sid, model)
        break

if not found:
    ph.closePort()
    sys.exit(
        "Nao encontrei nenhum servo.\n"
        " - liga so 1 servo ao U2D2\n"
        " - confirma a alimentacao externa dos servos (LED dá flash ao ligar)\n"
        " - confirma o cabo de 3 pinos (TTL) bem encaixado dos dois lados"
    )

baud, sid, model = found
print(f"Servo encontrado: ID atual = {sid}, model = {model}, @ {baud} baud")

if target_id is None:
    print("\n(so leitura — passa um ID de destino para mudar, ex.: "
          "venv/bin/python3 scripts/set_servo_id.py 3)")
    ph.closePort()
    sys.exit(0)

# 2) Desligar torque antes de escrever na EEPROM
pk.write1ByteTxRx(ph, sid, ADDR_TORQUE_ENABLE, 0)

# 3) Mudar o ID
if sid != target_id:
    comm, err = pk.write1ByteTxRx(ph, sid, ADDR_ID, target_id)
    if comm != COMM_SUCCESS or err != 0:
        ph.closePort()
        sys.exit(f"Falhou escrever o ID (comm={comm}, err={err})")
    print(f"ID: {sid} -> {target_id}  OK")
    sid = target_id
else:
    print(f"ID ja era {target_id}, nada a fazer")

# 4) Forcar baudrate 57600
if baud != TARGET_BAUD:
    comm, err = pk.write1ByteTxRx(ph, sid, ADDR_BAUD, BAUD_CODE_57600)
    if comm != COMM_SUCCESS or err != 0:
        print(f"AVISO: falhou por o baudrate a 57600 (comm={comm}, err={err}) "
              f"- ajusta na Dynamixel Wizard")
    else:
        print(f"Baudrate: {baud} -> 57600  OK")
else:
    print("Baudrate ja era 57600")

# 5) Confirmar
ph.setBaudRate(TARGET_BAUD)
model2, comm, err = pk.ping(ph, target_id)
if comm == COMM_SUCCESS:
    print(f"\nConfirmado: servo responde agora como ID {target_id} @ 57600. "
          f"Podes ligar o proximo.")
else:
    print(f"\nAVISO: nao confirmei o servo em ID {target_id} @ 57600 "
          f"(comm={comm}). Verifica na Wizard.")
ph.closePort()
