import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge
from tqdm import trange, tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class calculateBoundary:

	def __init__(self, grid):
		self._grid_bounds=grid[grid[:,3]==1]
		self._grid_inside=grid[grid[:,3]==0]

	def run(self, name="BayesianRidge"):
		if self._grid_bounds.size>0 and self._grid_inside.size>0:

			X=self._grid_inside[:,:2]
			y=self._grid_inside[:,2]

			sc_X= StandardScaler()
			X=sc_X.fit_transform(X)	
			model=0
			if name=="BayesianRidge": model= BayesianRidge(compute_score=True)
			if name=="LinearRegression": model= LinearRegression()
			if name=="SVR": model= SVR(kernel='poly', C=1000, gamma='auto', degree=3, epsilon=.1, coef0=1)

			model.fit(X, y)
			z_axis=model.predict(sc_X.transform(self._grid_bounds[:,:2]))
			self._grid_bounds[:,2]=z_axis
	def getGrid(self):	
		return np.concatenate((self._grid_bounds, self._grid_inside), axis=0)

path= "./topo.txt"
gridSize = 500
data=np.loadtxt(path, delimiter=' ')

data[:,:3]=data[:,:3]
maxvalues=np.max(data, axis=0)[:3]
minvalues=np.min(data, axis=0)[:3]
grid_values= maxvalues-minvalues
x_grids, y_grids, _=(np.ceil(grid_values/gridSize)).astype(np.int32)
test=[]
for x_count in trange(x_grids):
	for y_count in range(y_grids):
		range1=minvalues[:2]+[gridSize*x_count, gridSize*y_count]
		range2=minvalues[:2]+[gridSize*(x_count+1), gridSize*(y_count+1)]
		grid=data[np.all(data[:,:2]>=range1[:2], axis=1) & np.all(data[:,:2]<range2[:2], axis=1)]
		if grid.size>0:
			calcBound=calculateBoundary(grid)
			calcBound.run("SVR")
			grid=calcBound.getGrid()
			with open("SVR.txt", "a") as myfile:
				np.savetxt(myfile, grid[:,:3] ,fmt="%.4f,%.4f,%.4f")
