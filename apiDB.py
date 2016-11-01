# -*- coding: utf-8 -*-
'''
Created on 04/07/2013

@author: Carlos Botello
'''
import MySQLdb as mdb

class DB(object):

    def __init__(self, nombrebd="epi", user="root", passwd=""):
#         self.db = mdb.connect(host="mysql.myprodu.com",user="carlos",passwd="siempre1",db=nombrebd)
#         self.db = mdb.connect(host="mysql.gtienda.com",user="carlos",passwd="siempre1",db=nombrebd)
        self.db = mdb.connect(host="localhost",user="root",passwd="1234",db=nombrebd)
        self.c = self.db.cursor(mdb.cursors.DictCursor)
        
    def close(self):
        self.commit()
        self.c.close()
        self.db.close()
        
    def commit(self):
        self.db.commit()

    def exe(self, sql):
        self.c.execute(sql)
        return [row for row in self.c.fetchall()]

    def exe1(self, sql):
        self.c.execute(sql)
        rows = self.c.fetchall()
        self.cierra()
        return [row for row in rows]

    def LastID(self):
        rows = self.Ejecuta("select last_insert_id() as ID")
        r = rows[0]["ID"]
        return r
    
    def escape_string(self, s):
        return mdb.escape_string(s)