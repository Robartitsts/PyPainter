import serial

class ABBRunner():

	def __init__(self, width, height):
		self.ser = None
		self.connected = False
		self.width = width
		self.height = height

	def sendCoord(self, x, y):
		if not self.connected:
			return False

		msg = "COORD:X:" + str(x) + ",Y:" + str(y) + ";"
		return self.sendSerial(msg)

	def next(self):
		if not self.connected:
			return False
		return self.sendSerial("NEXT;")

	def end(self,):
		if not self.connected:
			return False
		return self.sendSerial("END;")

	def decidePaint(self, col):
		if not self.connected:
			return False

		msg = "SWAP:" + col + ";"
		return self.sendSerial(msg)

	def sendCanvasInfo(self):
		if not self.connected:
			return False

		# Wait for robot to be ready
		if not self.waitRobotReady():
			return False

		msg = "SIZE:X:" + str(self.width) + ",Y:" + str(self.height) + ";"
		return self.sendSerial(msg)

	def setSize(self, width, height):
		self.width = width
		self.height = height

	def waitRobotReady(self):
		for i in range(60):
			code = self.readSerialLine()
			print("waiting on robot...")
			if code:
				print(code)
				return True

		return false

	def abort(self,):
		self.ser.close()

	def connectToSerial(self, port):
		self.ser = serial.Serial(port, 115200, timeout=1)
		self.connected = True

	def readSerial(self,):
		msg = self.ser.read(2)
		return msg

	def readSerialLine(self,):
		msg = self.ser.readline()
		return msg

	def sendSerial(self, msg):
		if self.connected == False:
			return False

		# TODO: Try/catch?
		self.ser.write(msg)
		return True