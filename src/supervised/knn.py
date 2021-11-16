from math import sqrt
# calculate the Euclidean distance between two vectors
class knn:
	def __init__(self):
		self.txt=""
	def euclidean_distance(self,row1, row2):
		distance = 0.0
		for i in range(len(row1)-1):
			distance += (row1[i] - row2[i])**2
			tot_dist=sqrt(distance)
			self.txt+=f"calculating distance with points {row1[i]} and {row2[i]} for euclidean distance of: {tot_dist}\n"
		return sqrt(distance)
	
	def get_neighbors(self,train,test_row,num_neighbors):
		distances=list()
		for train_row in train:
			dist=self.euclidean_distance(test_row,train_row)
			distances.append((train_row,dist))
		self.txt+=f"\noriginal distances:{distances}\n"
		distances.sort(key=lambda tup:tup[1])
		self.txt+=f"\nwe will now sort the points based on there distances. sorted distances:{distances}\n"
		self.neighbors=list()
		for i in range(num_neighbors):
			self.neighbors.append(distances[i][0])
		self.txt+=f"\n{self.neighbors}"
		return self.neighbors
	def predict(self,train,test_row,num_neighbors):
		neighbors=self.get_neighbors(train,test_row,num_neighbors)
		output_values=[row[-1] for row in neighbors]
		print([row[-1] for row in neighbors])
		self.txt+=f"\noutput values={output_values}"
		self.preds=max(set(output_values),key=output_values.count)
		return self.preds
# Test distance function
if __name__ == '__main__':
	dataset = [[2.7810836,2.550537003,0],
		[1.465489372,2.362125076,0],
		[3.396561688,4.400293529,0],
		[1.38807019,1.850220317,0],
		[3.06407232,3.005305973,0],
		[7.627531214,2.759262235,1],
		[5.332441248,2.088626775,1],
		[6.922596716,1.77106367,1],
		[8.675418651,-0.242068655,1],
		[7.673756466,3.508563011,1]]
	model=knn()
	prediction = model.predict(dataset, dataset[9], 5)
	print(model.txt)
	print('Expected %d, Got %d.' % (dataset[0][-1], prediction))