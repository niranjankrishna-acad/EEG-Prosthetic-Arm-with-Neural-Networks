import tensorflow as tf 
import numpy
import pickl

def test(X,hw,hb):
	nextInput = tf.add(tf.matmul(X,hw),hb)
	fc1 = tf.nn.relu(nextInput)
	prediction = tf.nn.softmax(fc1)
	return prediction
def close():
	#add servo movement here
	pass
def open():
	#add servo movement here
	pass
def move(prediction,a):
	with tf.Session() as sess:
		print(prediction.eval())
		print(np.round(prediction.eval()))
		if(prediction.eval()>0.5):
			if(a==0):
				close()
				a = 1
			elif(a==1):
				open()
				a = 0


X = [[1000,352,2624,24672,1365,136,13616]]
pickle_in = open("Model.dict","rb")
cnndict = pickle.load(pickle_in)
prediction = test(X,cnndict["hw"],cnndict["hb"])
