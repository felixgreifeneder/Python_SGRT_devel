import untangle
import urllib

scihub_results = untangle.parse('/raid0/sgrt_config/query_results_israsel.txt')

entrylist = scihub_results.feed.entry
ids = list()

for entry in entrylist:
    ids.append(entry.id.cdata.encode().strip())

urlconnection = urllib.URLopener()
for id in ids:
    durl = "https://scihub.copernicus.eu/dhus/odata/v1/Products('" + id + "')/$value"
    urlconnection.retrieve()