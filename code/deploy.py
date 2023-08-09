from http.server import HTTPServer,BaseHTTPRequestHandler
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import pandas

host = "localhost"
port = 80
temperatureModele = pickle.load(open('./TrainedModele/temperatureModele.pkl','rb'))
temperatureSS = pickle.load(open('./TrainedModele/temperatureSS.pkl','rb'))
P0Modele = pickle.load(open('./TrainedModele/P0Modele.pkl','rb'))
P0SS = pickle.load(open('./TrainedModele/P0SS.pkl','rb'))
humidityModele = pickle.load(open('./TrainedModele/humidityModele.pkl','rb'))
humiditySS = pickle.load(open('./TrainedModele/humiditySS.pkl','rb'))

class HttpTraitement(BaseHTTPRequestHandler):
    def do_GET(self):
        #['annee','mois','jour','heure','lon','lat','humidity','P0','P1','P2','temperature']
        # localhost/target=temperature/1/annee=2018/2/mois=10/3/jour=01/4/heure=0.2/5/lon=36.781/6/lat=-1.291/7/humidity=47.9/8/P0=5/9/P1=11.4/10/P2=7.68
        url = self.path
        result = ''
        target = url[url.index('target')+7:url.index('/1/')]
        if target == 'temperature':
            annee = url[url.index('annee')+6:url.index('/2/')]
            mois = url[url.index('mois')+5:url.index('/3/')]
            jour = url[url.index('jour')+5:url.index('/4/')]
            heure = url[url.index('heure')+6:url.index('/5/')]
            lon = url[url.index('lon')+4:url.index('/6/')]
            lat = url[url.index('lat')+4:url.index('/7/')]
            humidity = url[url.index('humidity')+9:url.index('/8/')]
            P0 = url[url.index('P0')+3:url.index('/9/')]
            P1 = url[url.index('P1')+3:url.index('/10/')]
            P2 = url[url.index('P2')+3:len(url)]

            df = pandas.DataFrame({'annee':[float(annee)],'mois':[float(mois)],'jour':[float(jour)],'heure':[float(heure)],'lon':[float(lon)],'lat':[float(lat)],'humidity':[float(humidity)],'P0':[float(P0)],'P1':[float(P1)],'P2':[float(P2)]})

            X = temperatureSS.transform(df)

            result = str(temperatureModele.predict(X)[0][0])
            print(result)

        try:
            data = result
            self.send_response(200)
            self.send_header("Content-type", "text/html")
        except:
            data = "Not Found"
            self.send_response(404)
        self.end_headers()
        self.wfile.write(bytes(data,'utf-8'))

server = HTTPServer((host,port), HttpTraitement)
print('Server running...')
server.serve_forever()