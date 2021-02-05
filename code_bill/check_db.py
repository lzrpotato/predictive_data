import sqlite3 as lite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_db():
    con = lite.connect('status_param.db')
    cur = con.cursor()
    cur.execute('''
        select exp,splittype,dnn,aux,maxlen,pretrain,avg(acc),count(acc) from results group by exp,maxlen,splittype,dnn,pretrain,aux order by exp,splittype,dnn,aux,pretrain,maxlen;
    ''')
    result = cur.fetchall()
    con.close()
    
    df_res = pd.DataFrame(result,columns=['exp','splittype','dnn','aux','maxlen','pretrain','acc','count'])
    print(df_res)
    return df_res

def read_all_db():
    con = lite.connect('status_param.db')
    cur = con.cursor()
    cur.execute('''
        select * from results order by exp,maxlen,splittype,dnn,pretrain,aux,fold;
    ''')
    result = cur.fetchall()
    con.close()
    df_res = pd.DataFrame(result,columns=['exp','maxlen','splittype','dnn','pretrain','aux','fold','acc','c1',
            'c2','c3','c4','StopEpoch','BestEpoch'])
    print(df_res)

def draw_all():
    df_res = check_db()
    df_res['dnn_aux'] = df_res['dnn'] + '_' + df_res['aux'].astype(str) 
    #print(df_res['dnn_aux'])
    #print(df_res[df_res['dnn_aux'].str.startswith('CNN_test3')])
    print(df_res[(df_res['exp'] == 19) & (df_res['dnn_aux'].str.startswith('CNN'))])
    for exp, df in df_res.groupby(['exp','splittype']):
        for p, df_p in df.groupby(['pretrain']):
            fig, ax = plt.subplots()
            #for a, ndf in df_p.groupby(['dnn_aux']):
            print(exp)
            
            #print(a)
            nn = df_p.pivot(index='maxlen',columns='dnn_aux',values=['acc'])
            ndf = df_p.loc[(df_p['dnn_aux'].str.startswith('CNNRes_32')) | (df_p['dnn_aux'].str.startswith('CNNAVG_32')) | \
                 (df_p['dnn_aux'].str.startswith('CNN_test3_0')) | (df_p['dnn_aux'].str.startswith('CNN_test3_1')) |
                 (df_p['dnn_aux'].str.startswith('CNNOri')) | (df_p['dnn_aux'].str.startswith('CNNAVG')) | 
                 (df_p['dnn_aux'].str.startswith('CNNFIX')) | (df_p['dnn_aux'].str.startswith('CNN')) ,:]
            #print(ndf)
            #print(nn)
            #print(ndf)
            sns.lineplot(x='maxlen',y='acc',hue='dnn_aux',data=ndf,ax=ax, marker="o")
            ax.set_title(f'{exp[1]}-{p} accuracy vs max tree length')
            plt.savefig(f'./results/line-{exp[0]}-{exp[1]}-{p}.png')
            plt.close()
        #print(df[['splittype','dnn','maxlen','pretrain','acc']])
    #plt.subplots()

def find_the_best():
    df_res = check_db()
    df_res['dnn_aux'] = df_res['dnn'] + '_' + df_res['aux'].astype(str) 
    for exp, df in df_res.groupby(['exp','splittype']):
        for p, df_p in df.groupby(['pretrain']):
            print(df_p.groupby('maxlen')['acc'].count())
            print(df_p.groupby('maxlen')['acc'].idxmax())
            print(df_p.loc[df_p.groupby('maxlen')['acc'].idxmax(),:])

#check_db()
#read_all_db()
draw_all()
#find_the_best()