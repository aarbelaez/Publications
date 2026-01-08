


import sys
from math import *

file_le=sys.argv[1]
file_ps=sys.argv[2]
file_mn=sys.argv[3]

dist_data=sys.argv[4]
dist_data_MN=sys.argv[5]


mn_info=[map(float, line.split()[:3]) for line in open(file_mn).readlines()]
le_info=[map(float, line.split()[:4]) for line in open(file_le).readlines()[1:]]
ps_info=[map(float, line.split()[:2]) for line in open(file_ps).readlines()]

def MetroGetId(mn):
    return int(mn[0])

def LocalExchangeGetId(le):
    return int(le[3])

def euclidean_distance(pair1, pair2):
    x1 = round(pair1[0],0)/1000
    y1 = round(pair1[1],0)/1000
    x2 = round(pair2[0],0)/1000
    y2 = round(pair2[1],0)/1000
    return ( sqrt( pow( (x1 - x2), 2) + pow( (y1 - y2), 2) ) );

def edistMN2LE(mn, le):
    return euclidean_distance(mn[1:], le)
def edistLE2LE(le1, le2):
    return euclidean_distance(le1, le2)

def getPrim(le):
    return int(le[0])
def getSec(le):
    return int(le[1])

mn_data={}
for i in range(0, len(mn_info)):
    mn_data[i]=[]
##distance LE -> MN
f = open(dist_data_MN,'w');
#for le in ps_info:
tdist=0
for le_i in range(0, len(ps_info)):
    le = ps_info[le_i]
    prim=-1
    sec =-1
    #for mn in mn_info:
    ##print mn
    for mn_i in range(0, len(mn_info)):
        mn=mn_info[mn_i]
        if(mn[0] == le[0]):
            prim=mn
            dist=edistMN2LE(mn, le_info[le_i])
            tdist=tdist+dist
            f.write(str(MetroGetId(mn))+' '+str(le_i)+' '+str(dist)+' '+str(dist)+'\n')
            mn_data[mn_i].append( le_info[ le_i ]);
        if(mn[0] == le[1]):
            sec=mn
            dist=edistMN2LE(mn, le_info[le_i])
            tdist=tdist+dist
            f.write(str(MetroGetId(mn))+' '+str(le_i)+' '+str(dist)+' '+str(dist)+'\n')
            mn_data[mn_i].append( le_info[ le_i ]);
        if( prim!= -1 and sec!=-1 ):
            break
    if( prim == -1 or sec == -1):
        print 'Error with input prim: ',prim,' sec: ',sec,' local exchange: ',le,'\n'
        sys.exit(0)
print 'Trivial solution distance: ', tdist
f.close()
f = open(dist_data, 'w')
for mn_i in range(0, len(mn_info)):
    mn = mn_info[mn_i]
    les = mn_data[mn_i]
    for le_i in range(0, len(les)):
        for le_j in range(le_i, len(les)):
            dist=str( edistLE2LE( les[le_i], les[le_j] ) )
            f.write( str(LocalExchangeGetId(les[le_i]))+' '+str(LocalExchangeGetId(les[le_j]))+' '+dist+' '+dist+'\n')
f.close();



