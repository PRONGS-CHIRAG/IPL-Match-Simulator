
from __future__ import print_function
from pyspark import SparkContext
# $example on$
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from numpy import array

sc.stop()
if __name__ == "__main__":

	sc = SparkContext(appName="PythonDecisionTreeRegressionExample")
	sc.setLogLevel("ERROR")
		
	def createlabeledpoints(f):
		b1c=int(f[0])
		b1s=float(f[1])
		b1a=float(f[2])
		b2c=int(f[3])
		b2s=float(f[4])
		b2a=float(f[5])
		blc=int(f[6])
		blw=float(f[7])
		bla=float(f[8])
		r=int(f[9])
		w=int(f[10])
		return LabeledPoint(r,[b1c,b1s,b1a,b2c,b2s,b2a,blc,blw,bla])
	def createlabeledpoints2(f):
		b1c=int(f[0])
		b1s=float(f[1])
		b1a=float(f[2])
		b2c=int(f[3])
		b2s=float(f[4])
		b2a=float(f[5])
		blc=int(f[6])
		blw=float(f[7])
		bla=float(f[8])
		r=int(f[9])
		w=int(f[10])
		return LabeledPoint(w,[b1c,b1s,b1a,b2c,b2s,b2a,blc,blw,bla])
		
		
	data = sc.textFile("/home/anup/Downloads/hopeyoudontforwardthistoanyone/output.csv")
	csvdata=data.map(lambda x:x.split(","))
	trainingdata1=csvdata.map(createlabeledpoints)
	trainingdata2=csvdata.map(createlabeledpoints2)
	model1 = DecisionTree.trainRegressor(trainingdata1, categoricalFeaturesInfo={0:10,3:10,6:10},impurity='variance',maxDepth=14, maxBins=30)
	model2 = DecisionTree.trainRegressor(trainingdata2, categoricalFeaturesInfo={0:10,3:10,6:10},impurity='variance',maxDepth=14, maxBins=30)
	model1.save(sc, "runs")
	model2.save(sc, "wickets")
	sc.stop()
