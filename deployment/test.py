import json
import requests

endpoint_url = 'http://localhost:9696/predict'

cup_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Mocha_cup%2C_designed_by_Adolf_Flad%2C_made_by_KPM_Berlin%2C_1902%2C_porcelain%2C_1_of_6_-_Br%C3%B6han_Museum%2C_Berlin_-_DSC04094.JPG/640px-Mocha_cup%2C_designed_by_Adolf_Flad%2C_made_by_KPM_Berlin%2C_1902%2C_porcelain%2C_1_of_6_-_Br%C3%B6han_Museum%2C_Berlin_-_DSC04094.JPG'
plate_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Paper_Plate.jpg/455px-Paper_Plate.jpg'

data = [cup_url, plate_url]

result = requests.post(endpoint_url, json=data).json()

for url, preds in zip(data, result):
    
    print('Image:', url)
    print(json.dumps(preds, sort_keys=False, indent=4), end='\n\n')