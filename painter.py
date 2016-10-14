
import Tkinter as tk


class Painter(tk.Tk):

	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.width = 0
		self.height = 0
		self.label = tk.Label(text="BobRoss.jpg")
		self.label.pack(padx=10, pady=10)
		self.resize(600,600)

		# Shapes *shapes;
	 #	int *width, *height;
		# bool stuffshowing;

	 #	CommandWindow *commandWin;
	 #	RunLogic *logic;

	 #	width = new int(self, 600);
		# height = new int(self, 600);
		# Ava = new CytonRunner(self, width, height);
		# this->stuffshowing = false;
		# this->shapes = new Shapes(self, );
		# chappie = new ABBRunner(self, *width, *height, this->shapes);
		# Web = new Webcam(self, width, height);
		# this->showGUI(self, );

	 #	# AAAAAAAA
		# CytonRunner *Ava;
		# ABBRunner *chappie;

	def setDimensions(self, width, height):
		self.width = width
		self.height = height
		
	def showGUI(self, ):
  		#   	stuffshowing = true;
		# sketch = new Sketchpad(self, width, height, shapes, Ava, chappie, Web);
		
		# commandWin = new CommandWindow(self, shapes);
		
		# logic = new RunLogic(self, *width, *height, shapes, Ava, chappie);
		# logic->setParent(self, this);
		# sketch->show(self, );
		pass

	def save(self, projectLocation):
		pass
	def load(self, projectLocation):
		pass
	def loadRobot(self, robotLocation):
		pass
	def loadPhotoCanny(self, image, threshold, min_line_length, skip_freq):
		pass
	def loadPhotoKmeans(self, image, colorCount, minRegionSize, skip_freq):
		pass
	def newClicked(self, ):
		pass
	def resize(self, width, height):
		self.minsize(width,height)

	def murderousRampage(self, ):
		pass


if __name__ == "__main__":
	painter = Painter()	
	painter.mainloop()