import sqlite3 as lite
import os

class Status():
    def __init__(self):
        self.db_name = 'status_bert.db'
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
                insert into results (dnn,fold,SplitType,MaxLen,acc,c1,c2,c3,c4,BestEpoch) 
                values (?,?,?,?,?,?,?,?,?,?) 
                on conflict(dnn,fold,SplitType,MaxLen) 
                do update 
                    set acc=excluded.acc,c1=excluded.c1,c2=excluded.c2,c3=excluded.c3,c4=excluded.c4,BestEpoch=excluded.BestEpoch
                        where acc < excluded.acc;
                ''',
            (p['dnn'],int(p['fold']),p['SplitType'],p['MaxLen'],p['acc'],p['c1'],p['c2'],p['c3'],p['c4'],int(p['CurEpoch'])))
            
            cur.execute('''
                update results
                    set CurEpoch=?, ok=?
                        where dnn=? and fold=? and SplitType=? and MaxLen=?;
            ''', (int(p['CurEpoch']),p['ok'],p['dnn'],int(p['fold']),p['SplitType'],p['MaxLen']))
        
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
                select * from results where dnn=? and fold=? and SplitType=? and MaxLen=?;
                ''',
                (p['dnn'],p['fold'],p['SplitType'],p['MaxLen'])
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
    p = {'BestEpoch':-1,'CurEpoch':-1,'acc':-1,'c1':-1,'c2':-1,'c3':-1,'c4':-1,'dnn':'CNN','SplitType':'15_tv','fold':0,'MaxLen':100,'ok':False}
    ss.save_status(p)
    param = {'dnn':'CNN','SplitType':'15_tv','fold':0,'MaxLen':100}
    res = ss.read_status(param)
    print(res)