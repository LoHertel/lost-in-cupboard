# Deploy Kitchenware Prediction Service to Kubernetes

Prerequisites:
* Have Docker installed. Try to run `docker -v` to verify.
* Have kubectl installed. Try to run `kubectl version --client` to verify.
* For local deployment: 
    * Have Kind installed for running local Kubernetes clusters. Try to run `kind --version` to verify. [Installation instructions](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
* For deployment to remote cluster: 
    * Make sure that kubectl is connected to the cluster.

## Save Final Model

Keras saves the trained models in HDF5 format. We need to convert the model to the [SavedModel format](https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk) for using it in Tensorflow Serve.

If you haven't trained a model yourself, you could download the pretrained final model from [02-model-training.ipynb](../02-model-training.ipynb) with this bash command:
```bash
cd .. # switch to root folder of this project, if not there already
fileid="1VjnVxRqKFT2CsDAIVrC7jlVKpPu34rPw"
filename="model_final.h5"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o "models/${filename}"
```

Run the script for exporting a model:

```bash
cd deployment
python save_model.py ../models/model_final.h5 kitchenware-final-model
```

> **Note:** Use `cd deployment` to switch to the *deployment* folder to run this script successfully.


## Get Model Signature

Inside the model signature are the layer names for input and output defined. When we use tesorflow-serving for serving the model, we need to use the correct names of the layers.

Run the following CLI command to print the model signature definition:


```bash
saved_model_cli show --dir kitchenware-final-model --tag_set serve --signature_def serving_default
```

It will output somthing similar to this:
```
The given SavedModel SignatureDef contains the following input(s):
  inputs['input_4'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 150, 150, 3)
      name: serving_default_input_4:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['dense_3'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 6)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```

From the output above we see, that the model's input name is `input_4` and the model's ouput name is `dense_3`. The input is of shape `(-1, 150, 150, 3)` and the ouput `(-1, 6)`. The `-1` stands for the number of predictions, which are fed to the model at the same time.

The input and output name have to be added to the python file `gateway.py`.
You could run the following command in bash to add the INPUT_NAME and OUTPUT_NAME in the python file:
```bash
INPUT=input_4
OUTPUT=dense_3
sed -i "s/^\(INPUT_NAME = \).*$/\1'$INPUT'/gm" gateway.py
sed -i "s/^\(OUTPUT_NAME = \).*$/\1'$OUTPUT'/gm" gateway.py
```


## Build Tensorflow Serving Container and for the gateway

Build the docker container for tensorflow serving:
```bash
docker build -t kitchenware-tf-serving:1.0 -f image-model.dockerfile .
```

Build the docker container for the Flask gateway:
```bash
docker build -t kitchenware-gateway:1.0 -f image-gateway.dockerfile .
```


## For local deployment: Create Kind cluster

Create local Kubernetes cluster:
```bash
kind create cluster --name kind-cluster
```

Register images in Kind:
```bash
kind load docker-image kitchenware-tf-serving:1.0 --name kind-cluster
kind load docker-image kitchenware-gateway:1.0 --name kind-cluster
```

## For remote deployment: Upload container

Some Kubernetes services are among others AWS EKS, GCP GKE, and Azure AKS.

Upload the built containers `kitchenware-tf-serving:1.0` and `kitchenware-gateway:1.0` to the container registry, that your cluster is using.

Replace the image names in the Kubernetes yaml files in this folder: `kube-config/` with the names from the container registry.


## Deploy service to cluster

Deploy service to Kubernetes
```bash
kubectl apply -f kube-config
```

See if it is working:
```bash
kubectl get pod
```
The command prints something similar to:
```Output
NAME                                            READY   STATUS    RESTARTS   AGE
kitchenware-7f99986545-mzdm9                    1/1     Running   0          7s
tf-serving-kitchenware-model-58888bc949-rsc6q   1/1     Running   0          7s
```

## Test prediction service

Forward port from localhost to Kubernetes cluster:
```bash
kubectl port-forward service/kitchenware 9696:80
```

Test service by running this command in a second terminal window:
```bash
python test.py
```
You will see the prediction output for the specified images.

Close the second terminal window, switch back to the first terminal window and press `Ctrl+C` to stop the port forwarding.


## Remove deployment

Remove deployment
```bash
kubectl delete -f kube-config
```

## For local deployment: Remove Kind cluster

Remove the whole cluster:
```bash
kind delete cluster --name kind-cluster
```