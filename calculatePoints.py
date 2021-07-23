import numpy as np
import pandas as pd
import open3d as o3d
import os
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, ARDRegression
from tqdm import trange, tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer


def splitInGrid(path: str, model:str= "SVR", gridSize:int = 1000)->None:
	'''
	Split all points in smaller grids
	'''
	if os.path.exists(f'{model}.txt'):
		os.remove(f'{model}.txt')
	data=np.loadtxt(path, delimiter=' ')
	data=nearestRadiusPoints(data)
	maxvalues=np.max(data, axis=0)[:3]
	minvalues=np.min(data, axis=0)[:3]
	grid_values= maxvalues-minvalues
	x_grids, y_grids, _=(np.ceil(grid_values/gridSize)).astype(np.int32)

	print(f'x split:{x_grids} y split:{y_grids}')
	test=[]
	for x_count in trange(x_grids):
		for y_count in range(y_grids):
			range1=minvalues[:2]+[gridSize*x_count, gridSize*y_count]
			range2=minvalues[:2]+[gridSize*(x_count+1), gridSize*(y_count+1)]
			grid=data[np.all(data[:,:2]>=range1[:2], axis=1) & np.all(data[:,:2]<=range2[:2], axis=1)]
			if grid.size>0:
				calcBound=calculateBoundary(grid)
				calcBound.run(model)
				calcBound.getGrid()

def splitNearestNeighbours(path:str, model:str= "BayesianRidge", points:int = 1000)->None:
	if os.path.exists(f'{model}.txt'):
		os.remove(f'{model}.txt')
	data=np.loadtxt(path, delimiter=' ')
	data=nearestRadiusPoints(data)
	nearestPointsCount(data, points, model)
class calculateBoundary:

	def __init__(self, grid:np.array)->None:
		self._grid_bounds=grid[grid[:,3]==1]
		self._grid_inside=grid[grid[:,3]==0]
		

	def run(self, name:str="BayesianRidge")->None:
		self.name=name
		if self._grid_bounds.size>0 and self._grid_inside.size>0:
			# print(f'bound:{self._grid_bounds.size} inside:{self._grid_inside.size}')
			X=self._grid_inside[:,:2]
			y=self._grid_inside[:,2]

			sc_X= StandardScaler()
			X=sc_X.fit_transform(X)	
			model=0
			if name=="BayesianRidge": model= BayesianRidge(compute_score=True)
			elif name=="ARD": model= ARDRegression(compute_score=True)
			elif name=="LinearRegression": model= LinearRegression()
			elif name=="SVR": model= SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

			model.fit(X, y)
			z_axis=model.predict(sc_X.transform(self._grid_bounds[:,:2]))
			self._grid_bounds[:,2]=z_axis

	def getGrid(self)->None:
		if self._grid_bounds.size>0 and self._grid_inside.size==0:
			pass
		else:
			fullgrid= np.concatenate((self._grid_bounds, self._grid_inside), axis=0)
			with open(f"{self.name}.txt", "a") as myfile:
				np.savetxt(myfile, fullgrid[:,:3] ,fmt="%.2f,%.2f,%.2f")


def nearestPointsCount(data:np.array, points:int, name:str='BayesianRidge')->None:
	bounds=data[data[:,3]==1]
	inner=data[data[:,3]==0]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(inner[:,:3])
	pcd.paint_uniform_color([0.5, 0.5, 0.5])
	pcd_tree = o3d.geometry.KDTreeFlann(pcd)
	newbounds=bounds.copy()
	model=0
	if name=="BayesianRidge": model= BayesianRidge(compute_score=True)
	elif name=="ARD": model= ARDRegression(compute_score=True)
	elif name=="LinearRegression": model= LinearRegression()
	elif name=="SVR": model= SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

	for count, bound in enumerate(tqdm(bounds)):
		[k, idx, _] =pcd_tree.search_knn_vector_3d(bound[:3,np.newaxis], points)
		nearPoints=inner[idx[1:]]

		X=nearPoints[:,:2]
		y=nearPoints[:,2]

		sc_X= StandardScaler()
		X=sc_X.fit_transform(X)	
		

		model.fit(X, y)
		z_axis=model.predict(sc_X.transform([bound[:2]]))
		newbounds[count, 2]=z_axis

	fullgrid= np.concatenate((newbounds, inner), axis=0)
	np.savetxt(f"{name}.txt", fullgrid[:,:3] ,fmt="%.2f,%.2f,%.2f")

def nearestRadiusPoints(data:np.array)->np.array:
	bounds=np.where(data[:,3]==1)[0]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(data[:,:3])
	pcd.paint_uniform_color([0.5, 0.5, 0.5])

	pcd_tree = o3d.geometry.KDTreeFlann(pcd)
	for bound in bounds:		
		[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[bound], 200)
		#paint radius 10 blue
		np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
		
	for bound in bounds:
		#paint center point red
		pcd.colors[bound] = [1, 0, 0]
	# o3d.visualization.draw_geometries([pcd])

	color=np.asarray(pcd.colors)
	remove_indices=np.where(color[:,2]==1)[0]
	data = np.delete(data, remove_indices, axis=0)
	np.savetxt('test.txt', data, fmt="%.4f,%.4f,%.4f, %i")
	return data

def removeCluster(path, min_limit=100, gridSize:int = 1000):
	data=np.loadtxt(path, delimiter=' ')

	maxvalues=np.max(data, axis=0)[:3]
	minvalues=np.min(data, axis=0)[:3]
	grid_values= maxvalues-minvalues
	x_grids, y_grids, _=(np.ceil(grid_values/gridSize)).astype(np.int32)

	print(f'x split:{x_grids} y split:{y_grids}')
	test=[]
	fullgrid=[]
	for x_count in trange(x_grids):
		for y_count in range(y_grids):
			range1=minvalues[:2]+[gridSize*x_count, gridSize*y_count]
			range2=minvalues[:2]+[gridSize*(x_count+1), gridSize*(y_count+1)]
			grid=data[np.all(data[:,:2]>=range1[:2], axis=1) & np.all(data[:,:2]<=range2[:2], axis=1)]
			
			if grid.size>0:
	# 			newgrid=np.zeros((grid.shape[0], 6))
	# 			newgrid[:,:3]=grid[:,:3]
	# 			newgrid[:,3:]=np.random.randint(1, 255, 3)/255
	# 			fullgrid.extend(newgrid)
	# fullgrid=np.asarray(fullgrid)
	# pcd = o3d.geometry.PointCloud()
	# pcd.points = o3d.utility.Vector3dVector(fullgrid[:,:3])
	# pcd.colors=o3d.utility.Vector3dVector(fullgrid[:,3:])
	# o3d.visualization.draw_geometries([pcd])
				newgrid=np.zeros((grid.shape[0], 6))
				newgrid[:,:3]=grid[:,:3]
				newgrid[:,3:]=np.random.randint(1, 255, 3)/255
				grid_mean=grid[:,:3].mean(axis=0)
				mean_diff=grid[:,:3]-grid_mean
				mean_diff=np.abs(mean_diff)
				limit_indices=np.where(mean_diff[:,2]>=min_limit)[0]
				for limit_index in limit_indices:
					newgrid[limit_index, 3:]=[1, 0, 0]
				fullgrid.extend(newgrid)
	fullgrid=np.asarray(fullgrid)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(fullgrid[:,:3])
	pcd.colors=o3d.utility.Vector3dVector(fullgrid[:,3:])
	o3d.visualization.draw_geometries([pcd])


	# data_mean=data[:,:3].mean(axis=0)
	# mean_diff=data[:,:3]-data_mean
	# mean_diff=np.abs(mean_diff)
	# limit_indices=np.where(mean_diff[:,2]>=min_limit)[0]
	# pcd = o3d.geometry.PointCloud()
	# pcd.points = o3d.utility.Vector3dVector(data[:,:3])
	# pcd.paint_uniform_color([0.5, 0.5, 0.5])
	# for limit_index in limit_indices:
	# 	pcd.colors[limit_index]=[1, 0, 0]
	# o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
	#BayesianRidge, ARD, LinearRegression, SVR
	path= "../topo.txt"
	removeCluster(path, 250, 2000)
	# splitInGrid(path)
	# splitNearestNeighbours(path, 'SVR', 300)