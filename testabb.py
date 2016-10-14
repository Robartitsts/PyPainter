import abb
import time
import code

rob = abb.ABBRunner(240,248)
rob.connectToSerial("COM39")
print(rob.sendCanvasInfo())
time.sleep(5)
rob.decidePaint("A")
code.interact(local=locals())
rob.abort()
