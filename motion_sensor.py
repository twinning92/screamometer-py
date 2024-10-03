import board
import digitalio
import time

pir = digitalio.DigitalInOut(board.D13)
pir.direction = digitalio.Direction.INPUT

while True:
    print(pir.value)
    time.sleep(1)


