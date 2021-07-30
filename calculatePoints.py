import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


def splitNearestNeighbours(path:str, points:int = 1000)->None:
	data=np.loadtxt(path, delimiter=' ').astype(np.int32)

	newdata=nearestRadiusPoints(data)
	nearestPointsCount(data, newdata, points)


def fixInnerPoints(data):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(data[:,:3])
	cl, ind = pcd.remove_statistical_outlier(nb_neighbors=16,
                                                    std_ratio=0.2)
	mask = np.ones(len(data), np.bool)
	mask[ind] = 0
	# display_inlier_outlier(pcd, ind)
	return mask
def reconstruct(data, newdata):
	data[:,2]=newdata[:,0]
	stayind=np.where(newdata[:,1]!=1)[0]
	data=data[stayind]
	np.savetxt(f"test.txt", data[:,:3] ,fmt="%.2f,%.2f,%.2f")

def nearestPointsCount(data:np.array, newdata:np.array, points:int)->None:
	boundInd=np.where((newdata[:,1]==0) & (data[:,3]==1))[0]
	innerInd=np.where((newdata[:,1]==0) & (data[:,3]==0))[0]
	bounds=data[boundInd]
	inner=data[innerInd]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(inner[:,:3])
	pcd.paint_uniform_color([0.5, 0.5, 0.5])
	pcd_tree = o3d.geometry.KDTreeFlann(pcd)
	newbounds=bounds.copy()
	# if name=="BayesianRidge": model= BayesianRidge(compute_score=True)
	# elif name=="ARD": model= ARDRegression(compute_score=True)
	# elif name=="LinearRegression": model= LinearRegression()
	model= SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)

	for count, bound in enumerate(bounds):
		[k, idx, _] =pcd_tree.search_knn_vector_3d(bound[:3,np.newaxis], points)
		nearPoints=inner[idx[1:]]

		X=nearPoints[:,:2]
		y=nearPoints[:,2]

		sc_X= StandardScaler()
		X=sc_X.fit_transform(X)	
		

		model.fit(X, y)
		z_axis=model.predict(sc_X.transform([bound[:2]]))
		
		changeBoundEvaluationInd=boundInd[count]
		newdata[changeBoundEvaluationInd]=[z_axis, 3]

	removeInd=fixInnerPoints(inner)
	innerRemoveInd=innerInd[removeInd]
	newdata[innerRemoveInd,1]=1
	reconstruct(data, newdata)
	np.savetxt(f"result.txt", newdata ,fmt="%.2f,%i")

def nearestRadiusPoints(data:np.array)->np.array:
	bounds=np.where(data[:,3]==1)[0]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(data[:,:3])
	pcd.paint_uniform_color([0.5, 0.5, 0.5])

	pcd_tree = o3d.geometry.KDTreeFlann(pcd)
	for bound in bounds:		
		[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[bound], 100)
		#paint radius 10 blue
		np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
		
	for bound in bounds:
		#paint center point red
		pcd.colors[bound] = [1, 0, 0]

	color=np.asarray(pcd.colors)
	remove_indices=np.where(color[:,2]==1)[0]
	newdata=np.zeros((data.shape[0],2))
	newdata[:,0]=data[:,2]
	newdata[remove_indices,1]=1
	# data = np.delete(data, remove_indices, axis=0)
	# np.savetxt('test.txt', data, fmt="%.2f,%.2f,%.2f, %i")
	return newdata

if __name__ == '__main__':
	#BayesianRidge, ARD, LinearRegression, SVR
	print(os.getcwd())
	path="./topo.txt"
	if os.path.exists(path):
		splitNearestNeighbours(path, 300)
		print("Completed")
	else:
		print(f"Cant find file {path}")

