from urllib.request import urlretrieve
url = ('https://springernature.figshare.com/ndownloader/files/39841687')
file = 'input/synthetic_alerts.csv'
urlretrieve(url, file)