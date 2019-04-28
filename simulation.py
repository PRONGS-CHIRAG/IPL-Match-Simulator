from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from numpy import array
sc.stop()
if __name__ == "__main__":
	
	sc = SparkContext(appName="PythonDecisionTreeRegressionExample")
	sc.setLogLevel("ERROR")
	model1 = DecisionTreeModel.load(sc, "runs")
	model2 = DecisionTreeModel.load(sc, "wickets")
	batsmen_cluster={}
	bowler_cluster={}
	with open('/home/anup/Downloads/hopeyoudontforwardthistoanyone/cluster_batsmen.csv') as f:
		for line in f:
			ar=line.split(',')
			a=[]
			a.append(int(ar[0]))
			a.append(float(ar[3]))
			a.append(float(ar[4]))
			batsmen_cluster[ar[2]]=a
			
	with open('/home/anup/Downloads/hopeyoudontforwardthistoanyone/cluster_bowler.csv') as f:
		for line in f:
			ar=line.split(',')
			a=[]
			a.append(int(ar[0]))
			a.append(int(ar[3]))
			a.append(float(ar[4]))
			bowler_cluster[ar[2]]=a
	for i in  range(10):
		print()
	arrr=['2.csv']
	for l in arrr:
		with open('input/'+l,'r') as f:
			i=0
			for line in f:
				i=i+1
				if(i==2):
					T1=line.split(',')[0]
				if(i==3):
					T2=line.split(',')[0]
				if(i==6):
					SCORE1=int(line.split(',')[1])
				if(i==11):
					SCORE2=int(line.split(',')[1])
			
		T1=T1.ljust(30)
		T2=T2.ljust(30)
		print(T1,' ACTUAL SCORE:',SCORE1)
		print(T2,' ACTUAL SCORE:',SCORE2)
		if(SCORE1>SCORE2):
			won=T1
		elif(SCORE1<SCORE2):
			won=T2
		else:
			won="BOTH"
		print(won," won")
		t1score=0
		for z in range(1,3):
			with open('input/'+l,'r') as f:
				i=0
				for line in f:
					i=i+1
					if((i==4 and z==1) or (i==9 and z==2)):
						bat=line.split(',')
						bat=[x.strip('\n') for x in bat]
					if((i==5 and z==1) or (i==10 and z==2)):
						bowl=line.split(',')
						bowl=[x.strip('\n') for x in bowl]	
			A=bat.pop(0)
			B=bat.pop(0)
			score=0
			for i in range(1,len(bowl)+1):
				X=bowl.pop(0)
				bat1=batsmen_cluster[A]
				bat2=batsmen_cluster[B]
				bow=bowler_cluster[X]
				test=sc.parallelize([array(bat1+bat2+bow)])
				predictions=model1.predict(test)
				results=predictions.collect()
				runs=int(round(results[0]))
				predictions=model2.predict(test)
				results=predictions.collect()
				wickets=int(results[0])
				score=score+runs
				if(z==2 and score>t1score):
					break
				for i in range(wickets):
					if not A:
						break
					A=B
					B=bat.pop(0)
				if(runs%2!=0):
					t=A
					A=B
					B=t
			if(z==1):
				t1score=score
				print(T1,' PREDICTED SCORE',score)
			else:
				if(t1score>score):
					w=T1
				elif(t1score<score):
					w=T2
				else:
					w="BOTH"
				print(T2,' PREDICTED SCORE',score,' : ',w)
				print('\n\n')
	sc.stop()		
