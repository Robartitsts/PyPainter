import abb
import time
import code

rob = abb.ABBRunner(2530,2530)
rob.connectToSerial("COM39")
print(rob.sendCanvasInfo())
time.sleep(2)
# rob.decidePaint("A")
code.interact(local=locals())
rob.abort()
