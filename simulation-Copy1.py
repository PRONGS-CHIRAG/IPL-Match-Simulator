import csv
import string
import random
import pandas as pd
import numpy as np

pp_prob=pd.read_csv('/home/anup/bdProject/bd/pp_prob.csv')
batsman_cluster=pd.read_csv('/home/anup/bdProject/bd/batting_cluster.csv')
bowler_cluster=pd.read_csv('/home/anup/bdProject/bd/bowling_cluster.csv')
cluster_prob=pd.read_csv('/home/anup/bdProject/bd/clusters_prob.csv')

print(pp_prob.head())
print(batsman_cluster.head())
print(bowler_cluster.head())
print(cluster_prob.head())

def pro_fun(batsman,bowler):
    a=pp_prob[(pp_prob.batsman == batsman) & (pp_prob.bowler == bowler)]
    prob=[]
    flag=0
    x=a.empty is True
    if(a.empty is False):
        for i in a.values[0][2:]:
            prob.append(i)
        if(np.isnan(prob[-1])):
            flag=1
    if((flag==1) | (a.empty is True)):
        b=batsman_cluster[batsman_cluster.player_name==batsman]
        if(b.empty is False):
            batsman_cluster_no=b.values[0][2]
        else:
            batsman_cluster_no=random.randint(0,9)
        c=bowler_cluster[bowler_cluster.player_name==bowler]
        if(c.empty is False):
            bowler_cluster_no=c.values[0][2]
        else:
            bowler_cluster_no=random.randint(0,6)
        d=cluster_prob[(cluster_prob.bowler_cluster==bowler_cluster_no) & (cluster_prob.batsman_cluster==batsman_cluster_no)]
        if(flag==1):
            prob[-1]=d.values[0][-1]
        else:
            for i in d.values[0][2:]:
                prob.append(i)
    return prob

#match 55
t1bat=["RV Uthappa","G Gambhir","C Munro","MK Pandey","YK Pathan","JO Holder","Shakib Al Hasan","SA Yadav","AS Rajpoot", "SP Narine", "Kuldeep Yadav"]
t2bat=['S Dhawan','DA Warner ','NV Ojha ','Yuvraj Singh','KS Williamson','DJ Hooda','MC Henriques','KV Sharma', 'B Kumar','BB Sran',
 'Mustafizur Rahman']
t1bowl=["YK Pathan","AS Rajpoot","Shakib Al Hasan","SP Narine","AS Rajpoot","SP Narine","JO Holder","C Munro","JO Holder","Kuldeep Yadav","Shakib Al Hasan","Kuldeep Yadav","SP Narine","Kuldeep Yadav","Shakib Al Hasan","Kuldeep Yadav","Shakib Al Hasan","SP Narine","JO Holder","AS Rajpoot"]
t2bowl=["B Kumar","BB Sran","B Kumar","BB Sran","KS Williamson","DJ Hooda","KV Sharma","DJ Hooda","KV Sharma","MC Henriques","Mustafizur Rahman","MC Henriques","Mustafizur Rahman","KV Sharma","BB Sran","B Kumar","Mustafizur Rahman","BB Sran","Mustafizur Rahman","B Kumar"]

#innings1
bat=1
batsmen=[0,1]
innings1_run=[]
innings1_wicket=[]
score=[0,0]
wickets=[0,0]
prob={}
flag=0
for i in range(20):
    bowler= t2bowl[i]
    if(flag==1):
        break
    for j in range(6):
        if(flag==1):
            break
        bat_bowl = (t1bat[batsmen[0]],bowler)
        val = pro_fun(bat_bowl[0],bat_bowl[1])
        runs_prob = val[:6]
        cum_prob=0
        run=0
        if(bat_bowl[0] not in prob.keys()):
            prob[bat_bowl[0]]=val[-1]
        else:
            prob[bat_bowl[0]]*=val[-1]
        if(prob[bat_bowl[0]] < 0.05):
            wickets[0]+=1
            innings1_run.append(0)
            innings1_wicket.append((i*6)+j+1)
            if(wickets[0]==10):
                flag=1
            bat+=1
            batsmen[0]=bat
            
        else:
            rand=random.random()
            for y in range(6):
                cum_prob+=runs_prob[y]
                if(cum_prob > rand):
                    if(y==5):
                        run=6
                    else:
                        run=y
                    break
            innings1_run.append(run)
            score[0]+=run
            if(run==1 or run == 3):
                batsmen[0],batsmen[1]=batsmen[1],batsmen[0]
            
    batsmen[0],batsmen[1]=batsmen[1],batsmen[0]
    
#innings2
bat=1
batsmen=[0,1]
innings2_run=[]
innings2_wicket=[]
wickets[1]=0
score[1]=0
flag=0
prob1={}
for i in range(20):
    bowler= t1bowl[i]
    if(flag==1):
        break
    for j in range(6):
        if(flag==1):
            break
        bat_bowl = (t2bat[batsmen[0]],bowler)
        val = pro_fun(bat_bowl[0],bat_bowl[1])
        runs_prob = val[:6]
        cum_prob=0
        run=0
        if(bat_bowl[0] not in prob1.keys()):
            prob1[bat_bowl[0]]=val[-1]
        else:
            prob1[bat_bowl[0]]*=val[-1]
        if(prob1[bat_bowl[0]] < 0.05):
            wickets[1]+=1
            innings2_run.append(0)
            innings2_wicket.append((i*6)+j+1)
            if(wickets[1]==10):
                flag=1
            bat+=1
            batsmen[0]=bat
            
        else:
            rand=random.random()
            for y in range(6):
                cum_prob+=runs_prob[y]
                if(cum_prob > rand):
                    if(y==5):
                        run=6
                    else:
                        run=y
                    break
            innings2_run.append(run)
            score[1]+=run
            if(score[1]>score[0]):
                flag=1
            if(run==1 or run == 3):
                batsmen[0],batsmen[1]=batsmen[1],batsmen[0]
            
    batsmen[0],batsmen[1]=batsmen[1],batsmen[0]

print('Innings 1 ')
print('Score / wicket')
print(score[0],'/',wickets[0])

print('Innings 2 ')
print('Score / wicket')
print(score[1],'/',wickets[1])
