import urllib.request
print("Start Downloading model..")
    
try:
    #test.zip
    url = "https://www.dropbox.com/s/94og5k3czc0j2of/final_test.zip?dl=1"
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close() 
    with open("test.zip", "wb") as f :
        f.write(data)
    print("End downloading.")
except:
    print("Download error.")