import tensorflow as tf
import numpy
import xlrd
import pickle
learning_Rate = 0.000001
iterations = 1000
epochs = 2
def preprocess(filename):
	loc = (filename)
	wb = xlrd.open_workbook(loc) 
	sheet = wb.sheet_by_index(0) 
	X = []
	Y = []
	for i in range(sheet.nrows):
		arr = []
		for j in range(sheet.ncols):
			if(j==sheet.ncols-1):
				Y.append(sheet.cell_value(i,j))
			else:
				arr.append(sheet.cell_value(i,j))
		X.append(arr)
	for i in range(len(X)):
		X[i] = numpy.asarray([X[i]])
	for i in range(len(Y)):
		text = Y[i].split(",")
		arr = []
		for j in range(len(text)):
			arr.append(int(text[j]))
		Y[i] = numpy.asarray([arr])
	return X,Y


	
	
X,Y = preprocess("data.xlsx")
X_shape = X[0].shape
Y_shape = Y[0].shape
Xp = tf.placeholder(tf.float64,shape=X_shape)
Yp = tf.placeholder(tf.float64,shape=Y_shape)

hw = tf.Variable(tf.truncated_normal([list(X_shape)[1], 2],dtype=tf.float64,stddev=1e-1))

hb = tf.Variable(tf.constant(1.0, shape=[2], dtype=tf.float64),trainable=True)

nextInput = tf.add(tf.matmul(Xp, hw), hb)

fc1 = tf.nn.relu(nextInput)
prediction = tf.nn.softmax(fc1)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Yp * tf.math.log(prediction), reduction_indices=[1]))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_Rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	prevAverageError = 0
	for i in range(iterations):
		print("Iteration "+str(i+1)+"/"+str(iterations))
		Error = 0
		for j in range(len(X)):
			sess.run(optimizer,feed_dict={Xp:X[j],Yp:Y[j]})
			error = sess.run(cross_entropy,feed_dict={Xp:X[j],Yp:Y[j]})
			Error+=error
			if(int((j+1)/epochs)==(j+1)/epochs):
				print("Epoch "+str((j+1)/epochs)+" - Error : "+str(error))
		Difference = prevAverageError-(Error/len(X))
		print("Average Error : "+str(Error/len(X))+" Difference: "+str(Difference))
		prevAverageError = (Error/(len(X)))
		Mdict = {"hw":hw.eval(),"hb":hb.eval()}
		pickle_out = open("Model.dict","wb")
		pickle.dump(Mdict,pickle_out)
		pickle_out.close()