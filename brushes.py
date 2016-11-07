import cv2, math
import numpy as np
import cairo
from PIL import Image
import random
from deap import creator, base, tools, algorithms
import time

def pilImageFromCairoSurface( surface ):

   cairoFormat = surface.get_format()
   if cairoFormat == cairo.FORMAT_ARGB32:
      pilMode = 'RGB'
      # Cairo has ARGB. Convert this to RGB for PIL which supports only RGB or
      # RGBA.
      argbArray = np.fromstring( surface.get_data(), 'c' ).reshape( -1, 4 )
      rgbArray = argbArray[ :, 2::-1 ]
      pilData = rgbArray.reshape( -1 ).tostring()
   else:
      raise ValueError( 'Unsupported cairo format: %d' % cairoFormat )
   pilImage = Image.frombuffer( pilMode,
         ( surface.get_width(), surface.get_height() ), pilData, "raw",
         pilMode, 0, 1 )
   pilImage = pilImage.convert( 'RGB' )
   return pilImage

def cv2ImageFromCairoSurface( surface, WIDTH, HEIGHT):

   cairoFormat = surface.get_format()
   if cairoFormat == cairo.FORMAT_ARGB32:
      pilMode = 'RGB'
      # Cairo has ARGB. Convert this to RGB for PIL which supports only RGB or
      # RGBA.
      argbArray = np.fromstring( surface.get_data(), 'c' ).reshape( -1, 4 )
      rgbArray = argbArray[ :, 2::-1 ]
      pilData = rgbArray.reshape( -1 )#.tostring()
   else:
      raise ValueError( 'Unsupported cairo format: %d' % cairoFormat )
   bgra = np.ndarray(buffer=surface.get_data(), dtype=np.uint8,
        shape=(WIDTH, HEIGHT, 4))
   # rgbArray = bgra[]

   # print(rgbArray)
   # cvImage = cv2.imdecode(rgbArray, cv2.IMREAD_COLOR )
   rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
   rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

   # pilImage = Image.frombuffer( pilMode,
   #       ( surface.get_width(), surface.get_height() ), pilData, "raw",
   #       pilMode, 0, 1 )
   # pilImage = pilImage.convert( 'RGB' )
   # print(rgb)
   return rgb

def imageMSE(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

class Stroke():
	def __init__(self, points, color, width):
		self.points = points
		self.color = color
		self.width = width

	def start(self):
		return self.points[0], self.points[1]

	def middle(self):
		return self.points[2], self.points[3]

	def end(self):
		return self.points[4], self.points[5]

	def getColor(self,):
		return self.color

	def getWidth(self):
		return self.width

def drawStroke(ctx, stroke):
	# ctx.translate (0.1, 0.1) # Changing the current transformation matrix
	# print(stroke.start())
	ctx.move_to(*stroke.start())
	# ctx.arc (0.2, 0.1, 0.1, -math.pi/2, 0) # Arc(cx, cy, radius, start_angle, stop_angle)
	# ctx.line_to (0.5, 0.1) # Line to (x,y)
	ctx.curve_to(*stroke.points) # Curve(x1, y1, x2, y2, x3, y3)
	ctx.set_source_rgb(*stroke.getColor()) # Solid color
	ctx.set_line_width(stroke.getWidth())
	ctx.stroke()

def makeRandomStroke():
	return Stroke(np.random.rand(6),np.random.rand(3),np.random.uniform(0,0.1))

def renderPic(strokeList):
	surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
	ctx = cairo.Context (surface)
	ctx.scale (WIDTH, HEIGHT) # Normalizing the canvas
	# pat = cairo.LinearGradient (0.0, 0.0, 0.0, 1.0)
	# pat.add_color_stop_rgba (1, 0.7, 0, 0, 0.5) # First stop, 50% opacity
	# pat.add_color_stop_rgba (0, 0.9, 0.7, 0.2, 1) # Last stop, 100% opacity
	ctx.set_source_rgb (1,1,1) # Solid color
	ctx.rectangle (0, 0, 1, 1) # Rectangle(x0, y0, x1, y1)
	ctx.fill()

	for stroke in strokeList:
		drawStroke(ctx, stroke)

	genImage = cv2ImageFromCairoSurface(surface, WIDTH, HEIGHT)
	return genImage

def renderSurface(strokeList):
	surface = cairo.ImageSurface (cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
	ctx = cairo.Context (surface)
	ctx.scale (WIDTH, HEIGHT) # Normalizing the canvas
	# pat = cairo.LinearGradient (0.0, 0.0, 0.0, 1.0)
	# pat.add_color_stop_rgba (1, 0.7, 0, 0, 0.5) # First stop, 50% opacity
	# pat.add_color_stop_rgba (0, 0.9, 0.7, 0.2, 1) # Last stop, 100% opacity
	ctx.set_source_rgb (1,1,1) # Solid color
	ctx.rectangle (0, 0, 1, 1) # Rectangle(x0, y0, x1, y1)
	ctx.fill()

	for stroke in strokeList:
		drawStroke(ctx, stroke)
	return surface

def randomChance():
	minerror = float('inf')
	minStrokeList = []
	for iteration in range(500):
		strokeList = []
		for k in range(4):
			strokeList.append(makeRandomStroke())
		genImage = renderPic(strokeList)
		mse = imageMSE(targetImage,genImage)
		if mse < minerror:
			minerror = mse
			minStrokeList = strokeList
			print(mse)

	finalImage = renderPic(minStrokeList)
	cv2.imwrite('brushestest.png',finalImage)
	print(imageMSE(targetImage,finalImage))
	# surface.write_to_png ("example.png") # Output to PNG





class GeneticStroke(Stroke):

	def mutGaussian(self, indpb=0.5, colpb=0.7,mu=0, sigma=0.1, muc=0, sigmac=10, maxdim=1):
		for k in range(len(self.points)):
			if np.random.uniform(0,1) <= indpb:
				self.points[k] = max(0, min(self.points[k] + random.gauss(mu,sigma),maxdim))

		for k in range(len(self.color)):
			if np.random.uniform(0,1) <= colpb:
				self.color[k] = max(0,min(self.color[k] + random.gauss(muc,sigmac),1))

		if np.random.uniform(0,1) <= indpb:
				self.width = max(0.04,min(self.width + random.gauss(mu,sigma*.05),.5))


def makeRandomGeneticStroke():
	return GeneticStroke(np.random.rand(6),np.random.rand(3),np.random.uniform(0,0.05))


def geneticGenerate(targetImage, numstrokes):
	# minerror = float('inf')

	# finalImage = renderPic(minStrokeList)
	# cv2.imwrite('brushestest.png',finalImage)
	# print(imageMSE(targetImage,finalImage))
	# surface.write_to_png ("example.png") # Output to PNG

	def evalAgent(individual):
		return [1./imageMSE(targetImage,renderPic(individual))]

	def mutStroke(individual, indpb=0.5, colpb=0.7, mu=0, sigma=0.1, muc=0, sigmac=30, maxdim=1):
		# individual
		# GeneticStroke.mutGaussian
		# print("mutating:",individual)
		for stroke in individual:
			stroke.mutGaussian(indpb,colpb,mu,sigma,muc,sigmac,maxdim)
		return [individual]

	# thing = [makeRandomGeneticStroke()]
	# print(thing[0].points,thing[0].color,thing[0].width)
	# mutStroke(thing,indpb=1)
	# print(thing[0].points,thing[0].color,thing[0].width)
	# return

	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	toolbox.register("attr_bool", random.uniform, -100, 100)
	toolbox.register("individual", tools.initCycle, creator.Individual, [makeRandomGeneticStroke], n=numstrokes)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	toolbox.register("evaluate", evalAgent)
	toolbox.register("mate", tools.cxOnePoint)
	toolbox.register("mutate", mutStroke, indpb=0.4, colpb=0.7, mu=0, sigma=0.1, muc=0, sigmac=0.05, maxdim=1)
	toolbox.register("select", tools.selTournament, tournsize=3)

	# tools.cxOnePoint(indpb)

	population = toolbox.population(n=100)
	print(population[0])
	print(population[1])

	# wow = [1,2,3]
	# wow[0] = mutStroke(population[0])
	# print("wow",wow)

	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)
	try:
		NGEN=10000
		for gen in range(NGEN):
			starttime = time.time()
			offspring = algorithms.varAnd(population, toolbox, cxpb=0.1, mutpb=0.8)
			fits = toolbox.map(toolbox.evaluate, offspring)
			for fit, ind in zip(fits, offspring):
				ind.fitness.values = fit
			top10 = tools.selBest(population, k=10)
			population = toolbox.select(offspring, k=len(population)-12) + top10 + toolbox.population(n=2)
			print("end iteration:",gen, time.time() - starttime, np.max(fits)*10000, np.min(fits)*10000, np.mean(fits)*10000)


	finally:

		for k in range(len(top10)):
			finalImage = renderPic(top10[k])
			cv2.imwrite('top10/'+str(k)+'.png',finalImage)
			surface = renderSurface(top10[k])
			surface.write_to_png('top10/surface'+str(k)+'.png') # Output to PNG
			print(top10[k][0].points,top10[k][0].color,top10[k][0].width)
			print(imageMSE(targetImage,finalImage))


# targetImage = cv2.imread('flower1.jpg')
targetImage = cv2.imread('square.jpg')

# WIDTH, HEIGHT = targetImage.shape[0],targetImage.shape[1]
dims = min(targetImage.shape[0],targetImage.shape[1])
WIDTH = dims
HEIGHT = dims
WIDTH = 64
HEIGHT = 64
print(WIDTH,HEIGHT)
targetImage = cv2.resize(targetImage, (WIDTH,HEIGHT)) 
cv2.imwrite('target.png',targetImage)

geneticGenerate(targetImage,5)
print(imageMSE(targetImage,targetImage))







