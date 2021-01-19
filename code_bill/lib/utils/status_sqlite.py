import sqlite3 as lite
import os

class Status():
    def __init__(self):
        self.db_name = 'status.db'
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
                insert into results (method,fold,dataset,acc,c1,c2,c3,c4,BestEpoch) 
                values (?,?,?,?,?,?,?,?,?) 
                on conflict(method,fold,dataset) 
                do update 
                    set acc=excluded.acc,c1=excluded.c1,c2=excluded.c2,c3=excluded.c3,c4=excluded.c4,BestEpoch=excluded.BestEpoch
                        where acc < excluded.acc;
                ''',
            (p['method'],int(p['fold']),p['dataset'],p['acc'],p['c1'],p['c2'],p['c3'],p['c4'],int(p['CurEpoch'])))
            
            cur.execute('''
                update results
                    set CurEpoch=?, ok=?
                        where method=? and fold=? and dataset=?;
            ''', (int(p['CurEpoch']),p['ok'],p['method'],int(p['fold']),p['dataset']))
        
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
                select * from results where method=? and fold=? and dataset=?;
                ''',
                (p['method'],p['fold'],p['dataset'])
            )
        
            res = cur.fetchone()
        except lite.Error as e:
            print('[read_status] error', e.args[0])

        finally:
            if conn:
                conn.close()
        
        return res