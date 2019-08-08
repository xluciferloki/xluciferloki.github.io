from PIL import Image
from PIL.ExifTags import TAGS
import os

output="../data/result.csv"
out=open(output,'a')
out.write('lat,lon\n')
fpath= '../../DataSet/SmartCity/scene1_jiading_lib_training'
for item in os.walk(fpath):
    ob=item[2]
    for i in ob:
        name=fpath+'/'+str(i)
        ret={}
        try:
            img=Image.open(name)
            if hasattr(img,'_getexif'):
                exifinfo=img._getexif()
                if exifinfo !=None:
                    for tag,value in exifinfo.items():
                        decoded=TAGS.get(tag,tag)
                        ret[decoded]=value
                        N1 = ret['GPSInfo'][2][0][0]
                        N2 = ret['GPSInfo'][2][1][0]
                        N3 = ret['GPSInfo'][2][2][0]
                        N=int(N1)+int(N2)*(1.0/60)+int(N3)*(1.0/360000)
                        E1 = ret['GPSInfo'][4][0][0]
                        E2 = ret['GPSInfo'][4][1][0]
                        E3 = ret['GPSInfo'][4][2][0]
                        E=int(E1)+int(E2)*(1.0/60)+int(E3)*(1.0/360000)
                        out.write(str(N)+','+str(E)+'\n')
        except:
            pass
out.close()
