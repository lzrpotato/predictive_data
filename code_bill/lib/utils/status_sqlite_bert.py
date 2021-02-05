import sqlite3 as lite
import os

class Status():
    def __init__(self):
        self.db_name = 'status_param.db'
        self.timeout = 10000
        lite.register_adapter(bool,int)
        lite.register_converter('bool',lambda v: int(v) != 0)

    def connect(self):
        conn = lite.connect(self.db_name,detect_types=lite.PARSE_DECLTYPES)
        return conn

    def save_status(self, p):
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % int(self.timeout))
            cur.execute('begin')
            cur.execute(
                '''
                insert into results (exp,maxlen,splittype,dnn,pretrain,aux,fold,acc,c1,c2,c3,c4,stopepoch,bestepoch) 
                values (?,?,?,?,?,?,?,?,?,?,?,?,?,?) 
                on conflict(exp,maxlen,splittype,dnn,pretrain,aux,fold) 
                do update 
                    set acc=excluded.acc,c1=excluded.c1,c2=excluded.c2,c3=excluded.c3,c4=excluded.c4
                        where acc < excluded.acc;
                ''',
            (
                p['exp'],p['maxlen'],p['splittype'],p['dnn'],p['pretrain'],p['aux'],
                int(p['fold']),p['acc'],p['c1'],p['c2'],p['c3'],p['c4'],p['stopepoch'],p['bestepoch'])
            )
            
            conn.commit()
        except lite.Error as e:
            print('[save_status] error', e.args[0])

        finally:
            if conn:
                conn.close()
    
    def read_status(self, p):
        res = None
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute('PRAGMA busy_timeout=%d' % (self.timeout))
            cur.execute(
                '''
                select * from results where dnn=? and fold=? and splittype=? and maxlen=?;
                ''',
                (p['dnn'],p['fold'],p['splittype'],p['maxlen'])
            )
        
            res = cur.fetchone()
        except lite.Error as e:
            print('[read_status] error', e.args[0])

        finally:
            if conn:
                conn.close()
        
        return res

if __name__ == '__main__':
    ss = Status()
    p = {'exp':1,'acc':-1,'c1':-1,'c2':-1,'c3':-1,'c4':-1,'dnn':'CNN','splittype':'15_tv','fold':0,'maxlen':100,'pretrain':'bert-base-cased','aux':True}
    ss.save_status(p)
    param = {'exp':1,'dnn':'CNN','splittype':'15_tv','fold':0,'maxlen':100,'pretrain':'bert-base-cased','aux':True}
    res = ss.read_status(param)
    print(res)