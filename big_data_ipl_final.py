import numpy as np
import pandas as pd
from numpy import array
from math import sqrt
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

from pyspark.sql.functions import col

spark.version
sqlContext = SQLContext(sc)
batting=sqlContext.read.csv("/home/anup/bdProject/bd/stats_batting_2.csv", header=True, mode="DROPMALFORMED")
bowling=sqlContext.read.csv("/home/anup/bdProject/bd/stats_bowling_1 .csv", header=True, mode="DROPMALFORMED")
bowling.show()
batting.show()

batting.na.drop()
bowling=bowling.na.drop()

FEATURES_COL=["runs_scored","balls_faced","times_out","batting_average","strike_rate"]
FEATURES_COL_BOWL=["runs_conceded","wickets_taken","overs_bowled","bowling_average","economy_rate","bowling_strike_rate"]

for col in batting.columns:
    if col in FEATURES_COL:
        batting = batting.withColumn(col,batting[col].cast('float'))
batting.show()

for col in bowling.columns:
    if col in FEATURES_COL_BOWL:
        bowling = bowling.withColumn(col,bowling[col].cast('float'))
bowling.show()

vecAssembler = VectorAssembler(inputCols=FEATURES_COL, outputCol="features")
df_kmeans = vecAssembler.transform(batting).select('player_name', 'features')
df_kmeans.show()

vecAssembler_bowl = VectorAssembler(inputCols=FEATURES_COL_BOWL, outputCol="features_bowl")
df_kmeans_bowl = vecAssembler_bowl.transform(bowling).select('player_name', 'features_bowl')
df_kmeans_bowl.show()

cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(df_kmeans) # requires Spark 2.0 or later

cost_bowl = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features_bowl")
    model = kmeans.fit(df_kmeans_bowl.sample(False,0.1, seed=42))
    cost_bowl[k] = model.computeCost(df_kmeans_bowl) # requires Spark 2.0 or later

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost_bowl[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost_bowl')

k = 10
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

k = 7
kmeans_bowl = KMeans().setK(k).setSeed(1).setFeaturesCol("features_bowl")
model_bowl = kmeans_bowl.fit(df_kmeans_bowl)
centers_bowl = model_bowl.clusterCenters()

print("Cluster Centers: ")
for center in centers_bowl:
    print(center)

transformed = model.transform(df_kmeans).select('player_name', 'prediction')
rows = transformed.collect()
print(rows[:3])

transformed = model_bowl.transform(df_kmeans_bowl).select('player_name', 'prediction')
rows_bowl = transformed.collect()
print(rows_bowl[:3])

df_pred = sqlContext.createDataFrame(rows)
df_pred.show()

df_pred_bowl = sqlContext.createDataFrame(rows_bowl)
df_pred_bowl.show()

bat_cluster=df_pred.toPandas()
bat_cluster.to_csv("/home/anup/bdProject/bd/batting_cluster.csv")

bowl_cluster=df_pred_bowl.toPandas()
bowl_cluster.to_csv("/home/anup/bdProject/bd/bowling_cluster.csv")

df_pred.show()
df_pred_bowl.show()

batting_prob=sqlContext.read.csv("/home/anup/bdProject/bd/r_prob.csv", header=True, mode="DROPMALFORMED")
bowling_prob=sqlContext.read.csv("/home/anup/bdProject/bd/wicket.csv",header=True,mode="DROPMALFORMED")

batting_prob_clus=batting_prob.join(df_pred_bowl,df_pred_bowl.player_name==batting_prob.bowler,'leftouter').select(batting_prob["*"],df_pred_bowl["prediction"])

batting_prob_clus=batting_prob_clus.selectExpr("_c0 as index","batsman as batsman","bowler as bowler","P0 as P0","P1 as P1","P2 as P2","P3 as P3","P4 as P4","P6 as P6","prediction as bowler_cluster")
batting_prob_clus=batting_prob_clus.join(df_pred,df_pred.player_name==batting_prob_clus.batsman,'leftouter').select(batting_prob_clus["*"],df_pred["prediction"])
batting_prob_clus=batting_prob_clus.selectExpr("index as index","batsman as batsman","bowler as bowler","P0 as P0","P1 as P1","P2 as P2","P3 as P3","P4 as P4","P6 as P6","bowler_cluster as bowler_cluster","prediction as batsman_cluster")


bowling_prob_clus=bowling_prob.join(df_pred_bowl,df_pred_bowl.player_name==bowling_prob.bowler,'leftouter').select(bowling_prob["*"],df_pred_bowl["prediction"])
bowling_prob_clus=bowling_prob_clus.selectExpr("_c0 as index","batsman as batsman","bowler as bowler","no_of_times as no_of_times","no_of_balls as no_of_balls","p_out as p_out","p_not_out as p_not_out","prediction as bowler_cluster")
bowling_prob_clus=bowling_prob_clus.join(df_pred,df_pred.player_name==bowling_prob_clus.batsman,'leftouter').select(bowling_prob_clus["*"],df_pred["prediction"])
bowling_prob_clus=bowling_prob_clus.selectExpr("index as index","batsman as batsman","bowler as bowler","no_of_times as no_of_times","no_of_balls as no_of_balls","p_out as p_out","p_not_out as p_not_out","bowler_cluster as bowler_cluster","prediction as batsman_cluster")
batting_prob_clus.show()

pd_bowling_prob=bowling_prob_clus.toPandas()
pd_batting_prob=batting_prob_clus.toPandas()

pd_batting_prob.to_csv("/home/anup/bdProject/bd/pd_batting_prob.csv")
pd_bowling_prob.to_csv("/home/anup/bdProject/bd/pd_bowling_prob.csv")

df=pd.read_csv('/home/anup/bdProject/bd/pd_batting_prob.csv')
df2=pd.read_csv('/home/anup/bdProject/bd/pd_bowling_prob.csv')

res = df.groupby(["bowler_cluster","batsman_cluster"])['P0','P1','P2','P3','P4','P6'].mean().round(2).reset_index()
res2 = df2.groupby(["bowler_cluster","batsman_cluster"])['p_not_out'].mean().round(2).reset_index()

ref = df.groupby(["batsman","bowler"])['P0','P1','P2','P3','P4','P6'].mean().round(2).reset_index()
ref2 = df2.groupby(["batsman","bowler"])['p_not_out'].mean().round(2).reset_index()

final_ref=ref
final_ref=final_ref.merge(ref2,how='outer',left_on=["batsman","bowler"],right_on=["batsman","bowler"])

final=res
final['p_not_out']=res2['p_not_out']

final_ref.to_csv('/home/anup/bdProject/bd/pp_prob.csv',index=None)
final.to_csv('/home/anup/bdProject/bd/clusters_prob.csv',index=None)        