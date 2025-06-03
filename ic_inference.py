import os
import sys
import time
import string
import random
import traceback

import torch
import torchvision

from ic_train import ImageClassification
from micromind.utils import parse_configuration
from multipart import parse_form_data, is_form_request

from wsgiref.simple_server import make_server
import json

import digitalhub as dh

class ImageClassification(ImageClassification):
    """Implements an image classification class for inference."""

    def forward(self, img):
        """Computes forward step for image classifier.

        Arguments
        ---------
        batch : List[torch.Tensor, torch.Tensor]
            Batch containing the images and labels.

        Returns
        -------
        Predicted logits.
        """
        return self.modules["classifier"](img)

    def compute_loss(self, pred, batch):
        """Ignoring because it's inference."""
        pass

    def configure_optimizers(self):
        """Ignoring because it's inference."""
        pass


def top_k_accuracy(k=1):
    """
    Computes the top-K accuracy.

    Arguments
    ---------
    k : int
       Number of top elements to consider for accuracy.

    Returns
    -------
        accuracy : Callable
            Top-K accuracy.
    """

    def acc(pred, batch):
        if pred[1].ndim == 2:
            target = pred[1].argmax(1)
        else:
            target = pred[1]
        _, indices = torch.topk(pred[0], k, dim=1)
        correct = torch.sum(indices == target.view(-1, 1))
        accuracy = correct.item() / target.size(0)

        return torch.Tensor([accuracy]).to(pred[0].device)

    return acc


def init(context, model_name:str, data_dir:str, conf_name:str):
    """Initializes the inference context with the model and configurations."""
    print(f"Initializing inference context for {model_name}")

    #project = context.project
    project = dh.get_or_create_project("micromind-image-classification")

    if(data_dir.endswith("/")):
        data_dir = data_dir.rstrip(data_dir[-1])
    setattr(context, "data_dir", data_dir)

    #hparams = parse_configuration("/shared/cfg/image_classification/" + conf_name)
    hparams = parse_configuration("cfg/image_classification/" + conf_name)

    # file temporary path
    try:
        os.mkdir(data_dir + "/upload")
        print("create dir data/upload")
    except OSError as error:
        print(f"create dir data/upload error:{error}")

    #download model
    try:
        os.mkdir(data_dir + "/trained_model")
        print("create dir data/trained_model")
    except OSError as error:
        print(f"create dir data/trained_model error:{error}")

    model = project.get_model(model_name)
    model_path = model.download(destination=data_dir + "/trained_model")

    hparams.ckpt_pretrained = model_path
    mind = ImageClassification(hparams=hparams)
    mind.eval()

    print(f"init done:{mind}")
    setattr(context, "hparams", hparams)
    setattr(context, "mind", mind)


def inference(mind, hparams, image_filename):
    img = torchvision.io.read_image(image_filename)
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=hparams.input_shape[1:]),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    img = preprocess(img.float() / 255)
    logits = mind(img[None])

    # Calcola le probabilità con softmax usando PyTorch
    probs = torch.softmax(logits, dim=1)

    # Ottieni le top 5 classi e le loro probabilità
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)

    # Stampa le top 5 classi e le loro probabilità
    response = []
    for idx, prob in zip(top5_indices[0], top5_probs[0]):
        response.append((idx.item(), round(prob.item(), 4)))
    return response


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def serve_multipart(context, event):
    try:
        content_type = event.headers.get('Content-Type', '')
        context.logger.info(f"Received multipart event: {content_type}")
        result = {}
        #environ = io.BytesIO(event.body)
        #environ = event.body
        if 'multipart/form-data' in content_type:
            context.logger.info("serve multipart buffer")
            environ = {
                "wsgi.input": io.BytesIO(event.body),
                "CONTENT_LENGTH": str(len(event.body)),
                "CONTENT_TYPE": content_type,
                "REQUEST_METHOD": "POST",
            }            
            forms, files = parse_form_data(environ)
            context.logger.info("serve multipart files")
            for filed_name in files:
                file_details = files[filed_name]
                context.logger.info(f"process file:{file_details.filename}")
                filename = context.data_path + "/upload/" + id_generator() + "_" + file_details.filename
                file_details.save_as(filename)
                context.logger.info(f"filename:{filename}") 

                classification_result = []
                result[filed_name] = classification_result
                for idx, prob in inference(context.mind, context.hparams, filename):
                    info = {}
                    info['class'] = idx
                    info['probability'] = prob
                    classification_result.append(info)

                if os.path.exists(filename):
                    os.remove(filename)
        
        return result
    except Exception as e:
        context.logger.error(f"serve_multipart error:{e}")
        return context.Response(body=f"Error:{e}", status_code=500)


def simple_app(environ, start_response):
    result = {}
    if is_form_request(environ):
        forms, files = parse_form_data(environ)
        for filed_name in files:
            try:
                file_details = files[filed_name]
                print(f"process file:{file_details.filename}")
                filename = "/home/nori/data/upload/" + id_generator() + "_" + file_details.filename
                file_details.save_as(filename) 

                classification_result = []
                result[filed_name] = classification_result
                for idx, prob in inference(main_context.mind, main_context.hparams, filename):
                    info = {}
                    info['class'] = idx
                    info['probability'] = prob
                    classification_result.append(info)

                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                traceback.print_exc()
                print(e)

    status = '200 OK'
    headers = [('Content-type', 'application/json; charset=utf-8')]
    content = json.dumps(result)
    content = [content.encode('utf-8')]
    start_response(status, headers)


class Context:
    data_dir = "/home/nori/data"
    hparams = None
    mind = None

if __name__ == "__main__":
    main_context = Context()
    init(main_context, "micromind-model-phinet", "/home/nori/data", "phinet.py")

    with make_server('', 8051, simple_app) as httpd:
        print("Serving on port 8051...")
        httpd.serve_forever()

    # Example usage