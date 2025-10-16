## Serving a model

1. Initializing the project
 ```Python
import digitalhub as dh

project = dh.get_or_create_project("micromind-image-classification")
```

2. Define a funtion that handle the serving process
```Python
func = project.new_function(
    name="inference_function",
    kind="python",
    python_version="PYTHON3_10",
    code_src="git+https://github.com/tn-aixpa/micromind-adapter",
    handler="ic_inference:serve_multipart",
    init_function="init",
    requirements=["micromind", "timm==0.6.13"]
)
```

3. Run the serving function
```Python
run = func.run(
    action="serve",
    resources = {"mem":{"requests": "4Gi",}},
    init_parameters={"conf_name":"phinet.py", "data_dir":"/data", "model_name":"ic-phinet"},
    volumes=[
        {
            "volume_type":"persistent_volume_claim",
            "name": "micromind-ic",
            "mount_path": "/data",
            "spec": { "size": "5Gi" }
        }
    ]
)
```

The following parameters are used in the init phase:
- conf_name: the model conf that you want to use, could be one of 'phinet.py' or 'xinet.py'
- data_dir: path used to store the model training files
- model_name: the name of the model to use

4. In order to invoke the service, an HTTP form-multipart post request must be invoked
```Python
import requests

file_path = "test-image.jpg"

run.refresh()
url = "http://" + run.status.service['url']

with open(file_path, "rb") as file:
    files = {'file': file}
    response =  run.invoke(files=files, method='POST')

print(f"response code:{response.status_code}")
print(f"response body:{response.text}")
```

