import requests

url = "http://localhost:8051"
file_path = "test_img/viale.jpg"

with open(file_path, "rb") as file:
    files = {'file': file}
    response = requests.post(url, files=files)
    print(response.status_code)
    print(response.text)