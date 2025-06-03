## Traning a model

The training phase is based on the available model configuration integrated into the Micromind Toolkit

1. Initializing the project
 ```Python
import digitalhub as dh

project = dh.get_or_create_project("micromind-image-classification")
```

2. Define a funtion that handle the training process
```Python
train_fn = project.new_function(name="train_function",
                                kind="python",
                                python_version="PYTHON3_10",
                                code_src="git+https://github.com/tn-aixpa/micromind-adapter",
                                handler="ic_train:train",
                                requirements=["micromind", "timm==0.6.13"])
```                                              

3. Run the training function
```Python
train_fn.run(
    action="job", 
    parameters={"conf_name":"phinet.py", "dataset":"torch/cifar10", "data_dir":"/data", "epochs":50},
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

The following parameters are used:
- conf_name: the model conf that you want to use, could be one of 'phinet.py' or 'xinet.py'
- dataset: the dataset to use for training, for example torch/cifar10 or torch/cifar100 
- epochs: number of epochs 
- data_dir: path used to store the model training files

Once complete, the function log a new model for the project with name "ic-<model-type>".