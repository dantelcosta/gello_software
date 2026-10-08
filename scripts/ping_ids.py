"""Varre baudrates e IDs a procura de servos Dynamixel do GELLO.

Uso:
    venv/bin/python3 scripts/ping_ids.py [porta]

Default: /dev/cu.usbserial-FTBENXKL. Testa varios baudrates comuns; para
cada um faz ping aos IDs 0..20. Serve para descobrir se o adaptador esta
mesmo ligado ao barramento Dynamixel e a que velocidade.
"""

import sys

from dynamixel_sdk import PacketHandler, PortHandler

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-FTBENXKL"
BAUDS = [57600, 1000000, 2000000, 115200, 9600, 3000000, 4000000]
IDS = range(0, 21)

ph = PortHandler(PORT)
pk = PacketHandler(2.0)

if not ph.openPort():
    sys.exit(f"Falhou abrir a porta {PORT}")

print(f"Porta {PORT}\n")
achou_algo = False
for baud in BAUDS:
    if not ph.setBaudRate(baud):
        print(f"  (nao consegui por baud {baud})")
        continue
    encontrados = []
    for i in IDS:
        model, comm, err = pk.ping(ph, i)
        if comm == 0:
            encontrados.append((i, model, err))
    if encontrados:
        achou_algo = True
        print(f"@ {baud:>7} baud:")
        for i, model, err in encontrados:
            tag = "OK" if err == 0 else f"erro hw {err}"
            print(f"      ID {i:>2}  model {model}  {tag}")
    else:
        print(f"@ {baud:>7} baud: nada")

ph.closePort()
if not achou_algo:
    print(
        "\nNENHUM servo em NENHUM baudrate.\n"
        "=> o adaptador provavelmente NAO esta ligado ao barramento do GELLO,\n"
        "   ou o GELLO nao tem alimentacao."
    )
