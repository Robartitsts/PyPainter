import abb
import time
import code

rob = abb.ABBRunner(2400,2480)
rob.connectToSerial("COM39")
print(rob.sendCanvasInfo())
time.sleep(2)
# rob.decidePaint("A")
scale = 3
rob.sendCoord(250*scale,250*scale)
time.sleep(3.5)
rob.sendCoord(250*scale,750*scale)
time.sleep(3.5)

rob.sendCoord(750*scale,750*scale)
time.sleep(3.5)

rob.sendCoord(750*scale,250*scale)

time.sleep(3.5)
rob.sendCoord(250*scale,250*scale)
time.sleep(1)
# rob.abort()

code.interact(local=locals())
rob.abort()
