#coding: utf-8
#from scipy.spatial.distance import pdist, squareform
#from sklearn.cluster import DBSCAN
import pandas as pd
import datetime
import numpy as np
import re
#import haversine as hv
#import matplotlib.pyplot as plt
#from math import radians, cos, sin, asin, sqrt
#from sklearn.decomposition import PCA

#!/usr/bin/python
hostname = 'localhost'#'srv044.it4pme.fr'
username = 'postgres'#'geo4cast'
password = '111111'#LY+tpLRQA5lmXT//'
database = 'postgres'#geo4cast'
section_coyote = 'section_coyote_fr'
osm_section = 'zone_sections'
distance_range = 5#5#10
divisionValue = 100000

lon = 3.99351
lat = 49.24313
deltaCap = 45
headersection = 'id,latitude,longitude,heading,speed,ts,section'+'\n'
header = 'id,latitude,longitude,heading,speed,ts'+'\n'

f = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/result.csv", "w")
f.write(headersection)#with section assignment
f0 = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/result0.csv", "w")
f0.write(header)#no corresponding section
f2 = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/result2.csv", "w")
f2.write(header)#two corresponding sections
f1 = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/result1.csv", "w")
f1.write(headersection)#one corresponding section


# Simple routine to run a query on a database and print the results:
def doQuery(conn,df) :
    countNoSection = 0
    countOneSection = 0
    countTwoOrMoreSection = 0
    sectionIdTemp = ''
    cur = conn.cursor()
    print(len(df.index))

    for index,row in df.iterrows():
        info = re.sub(r"\s+", " ", (",".join(row.to_string(header=False,index=False).split("\n")))).replace(", ",",")
        lon = row['longitude']#current coordinate of vehicle
        print(lon)
        lat = row['latitude']#current coordinate of vehicle
        print(lat)
        cap = row['heading']#current heading of vehicle

        #WITH fastsort as (
        #    select
        #    *
        #    *
        #from
        #%(osm_section)s
        #where
        #%(osm_section)s.wkb_geometry && ST_MakeEnvelope(5.349222, 43.398268, 5.354393, 43.391657, 4326)
        #order by
        #st_point(%(lon)s, %(lat)s) <#> wkb_geometry
        #limit 15
        #)

        SQLquery = """ 
                        WITH result as (                            
                            SELECT
                                ogc_fid,
							    osm_id,
                                ST_distance(
                                    st_transform(st_setsrid( wkb_geometry::geometry , 4326), 2100),
                                    st_transform(st_setsrid (st_point(%(lon)s, %(lat)s), 4326), 2100)  
                                )  As distance_m,
                                ST_distance(
                                    st_transform(
                                        st_closestpoint(
                                            st_setsrid( wkb_geometry::geometry , 4326), 
                                            st_setsrid( st_point(%(lon)s, %(lat)s), 4326)
                                        ), 
                                        2100 
                                    ),
                                    st_transform(st_setsrid (st_point(%(lon)s, %(lat)s), 4326), 2100)  
                                )  As distance_cl_point,
                                other_tags,
                                st_asgeojson(wkb_geometry::geometry) as line_string,
                                st_npoints(wkb_geometry) as number_points,
                                round(
                                    st_azimuth(
                                        st_pointn(wkb_geometry, 1), 
                                        st_pointn(wkb_geometry, st_npoints(wkb_geometry))
                                    ) / (2 * pi()) * 360
                                ) as azimuth,
                                type
                            FROM
                                %(osm_section)s
                        )
                        SELECT
                            *
                        FROM
                            result
                        WHERE
                            distance_cl_point < %(distance_range)s
                        ORDER BY
                            distance_cl_point
    
                        """ % { 'osm_section': osm_section,
                                'lat' : lat,
                                'lon': lon,
                                'distance_range': distance_range
                                }
        cur.execute(SQLquery)#The query is to retrieve possible sections of the vehicle within the given distance range. Accordingly, it takes
        #vehicle coordinate as input, looks up wkb_geometry field of map data to see which sections statisfy the criterion. In addition, the heading
        #filter helps to more exactly pick the right lane up, in case there are two adjacent lanes with opposite headings.

        result = cur.fetchall()#list of corresponding sections of the coordinate

        # if no sections is found, increment noFoundDistanceFilterCounter
        if len(result) != 0:#at least one section found by distance range
            tempCount = 0;#count corresponding sections
            #coordinate = string[string.index("[")+2:string.index("]")]

            for index_result, line in enumerate(result, start=1):#pick one by one to check if that one is good
                azimuth = line[7]
                if azimuth != None:
                    # calculate the angle between cap and azimuth if no segment has been found yet
                    # get the angle between zap and road azimuth orientation
                    alpha = max([cap, azimuth]) - min([cap, azimuth])

                    # get the smallest angle for alpha
                    if alpha < 180:
                        diffCap = alpha
                    else:
                        diffCap = 360 - alpha
                    # 						logger.debug("diffCap: %s" %diffCap)

                    # if the angle is smaller than the value deltaCap we fix, accept the angle
                    if diffCap < deltaCap:#a corresponding section appears
                        sectionIdTemp = line[0]#then get section name
                        #for i in range(len(coordinateList)):
                            #distance = hv.haversine([lat, lon], [float(theCoordinate[1]), float(theCoordinate[0])])
                            #print(distance)
                            #if distance < minDistance:
                                #minDistance = distance
                                #minDistanceIndex = i
                        #print('min')
                        #print(minDistance)
                        #closestCoordinate = coordinateList[minDistanceIndex].split(',')
                        #df.set_value(index,'latitude',closestCoordinate[1])  #coordinateList[1]
                        #df.set_value(index,'longitude',closestCoordinate[0])  #coordinateList[1]
                        #print(df)

                        tempCount+=1#count corresponding section
                        if tempCount > 1:
                            countTwoOrMoreSection+=1
                            print('MORE THAN ONE SECTION')
                            f2.write(info+'\n')
                            f.write(info + ',S-2\n')
                            break#more than 1 section means the point is done as an invalid one, and we should move to next point.

                if(index_result) == len(result):
                    if tempCount == 1:
                            countOneSection+=1
                            print('ONE SECTION')#valid coordinate with only one section
                            f.write(info+',S-'+str(sectionIdTemp)+'\n')
                            f1.write(info+',S-'+str(sectionIdTemp)+'\n')

                    if tempCount == 0:
                            countNoSection+=1
                            print('NO SECTION ')#invalid coordinate with no section
                            #print('S-0')
                            f.write(info+',S-0\n')
                            f0.write(info+'\n')

        else:
            countNoSection+=1
            print('NO SECTION')
            f.write(info+',S-0\n')
            f0.write(info+'\n')
    #df2=df1.drop_duplicates(subset=['id', 'latitude','longitude'], keep='first')
    f.close()
    print('NO SECTION COUNT:')
    print(countNoSection)
    print('ONE SECTION COUNT:')
    print(countOneSection)
    print('MORE THAN ONE SECTION COUNT:')
    print(countTwoOrMoreSection)

print ("Using psycopg2…")
import psycopg2
begintotalTime = datetime.datetime.now()
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

np.set_printoptions(threshold=np.nan)
df=pd.read_csv("Paris_zone_weekday_May19-Feb21.csv",sep=',')

#byid = df.groupby('id')#.reset_index()
#stdbyid=byid.std()
#idlist=[]
#idlist=stdbyid[stdbyid['latitude']==0].index
#print(idlist)
#print_full(stdbyid)

#df1=df.drop_duplicates(subset=['id', 'latitude','longitude'], keep='first')
#dfcoords = pd.DataFrame(columns=['latitude', 'longitude'])
#for i in range(len(idlist)):
#    for index,row in df1.iterrows():
#        if row['id']==idlist[i]:
#            #dfcoords.iloc[i,'longitude'] = 1.0 #df1.at[j,'longitude']
#            #dfcoords.iloc[i,'latitude'] = 2.0  #df1.at[j,'latitude']
#            dfcoords.set_value(i,'longitude',row['longitude'])
#            dfcoords.set_value(i,'latitude',row['latitude'])

#print(dfcoords)
#df3=df1.drop('id', axis=1) this is for clustering on a set of coordinate and coordinate only
#df3=df1.drop('ts', axis=1)

myConnection = psycopg2.connect( host=hostname, user=username, password=password, dbname=database)#connect to get map data
doQuery( myConnection,df)#get map data
myConnection.close()
endtotalTime = datetime.datetime.now()
print('total processing Time')
processingTime = endtotalTime - begintotalTime #time complexity
print(processingTime.seconds)
