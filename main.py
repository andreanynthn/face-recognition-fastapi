from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
import os
import string
import numpy as np
import pandas as pd
from deta import Deta
from deepface import DeepFace
from pydantic import BaseSettings
from functools import lru_cache
from deepface.commons import functions
from os import path, remove, getcwd, makedirs


# model
model = DeepFace.build_model("Facenet512")

# image dir
tmp_dir = path.join(getcwd(), "tmp")
if not path.exists(tmp_dir):
    makedirs(tmp_dir)

# unknown image dir
unk_dir = path.join(getcwd(), "unk_dir")
if not path.exists(unk_dir):
    makedirs(unk_dir)

# function for retrieve names
def retrieveName():
    retrieve = db.fetch()
    results = retrieve.items

    instances = []
    list_name = []
    for i in range(len(results)):
        img_name = results[i]["img_name"]

        instance = []
        instance.append(img_name)
        list_name.append(img_name)
        instances.append(instance)

    name_df = pd.DataFrame(instances, columns = ["name"])
    name_df_dict = name_df.to_dict('list')

    return name_df, name_df_dict, list_name

# function for retrieve images
def retrieveImage():
    retrieve = db.fetch()
    results = retrieve.items

    instances = []
    for i in range(len(results)):
        img_name = results[i]["img_name"]
        embedding_bytes = np.array(results[i]["embedding"])
        embedding = np.array(embedding_bytes)

        instance = []
        instance.append(img_name)
        instance.append(embedding)
        instances.append(instance)

    result_df = pd.DataFrame(instances, columns = ["img_name", "embedding"])

    return result_df

# function for cosine distance
def findCosineDistance(df):
    vector_1 = df['embedding']
    vector_2 = df['target']
    a = np.matmul(np.transpose(vector_1), vector_2)

    b = np.matmul(np.transpose(vector_1), vector_1)
    c = np.matmul(np.transpose(vector_2), vector_2)

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

# function for cosine similarity
def findCosineSimilarity(df):
    vector_1 = df['embedding']
    vector_2 = df['target']
    a = np.matmul(np.transpose(vector_1), vector_2)

    b = np.matmul(np.transpose(vector_1), vector_1)
    c = np.matmul(np.transpose(vector_2), vector_2)

    similarity = a / (np.sqrt(b) * np.sqrt(c))
    perc_similarity = np.round(similarity*100, 2)

    return perc_similarity



# Fast API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://0.0.0.0:8080",
        "http://localhost:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Settings(BaseSettings):
    base_url: str = "http://127.0.0.1:8080"
    project_key: str = "project key"
    database: str = "db"

    class Config:
        env_file = ".env"


# setup database
settings = Settings()

deta = Deta(settings.project_key)
db = deta.Base(settings.database)

# @app.get("/")
# async def root():
#     return({"massage" : "Face Recognition"})


# get names
@app.get("/face-recognition/get-names")
def getName():
    return retrieveName()[1]

# registration image
@app.post("/face-recognition/input-image")
def inputImage(upload_file: UploadFile = File(), name: str = Form(description = "enter full name")):
    file_name = f"{name}.jpg"
    file_path = f"{tmp_dir}/{file_name}"
    with open(file_path, "wb+") as file_object:
        file_object.write(upload_file.file.read())
    # image = upload_file
    username = name
    facial_img = functions.preprocess_face(file_path, target_size = (160, 160), detector_backend = 'ssd')

    # embedding
    embedding = model.predict(facial_img)[0]

    # output
    output = {
        "registration status" : None
    }

    # insert image into database
    if db.put({
        "img_name" : username.lower(),
        "embedding" : embedding.tolist()
    }, key=name):
        output["registration status"] ="success"

    else:
        output["registration status"] = "unsuccessful"

    return output

# remove image
@app.delete("/face-recognition/remove-image")
def removeImage(name: str = Form(description="enter full name")):
    username = name.lower()

    # output
    output = {
        "remove status" : None
    }

    # check whether name is exist
    # if exist, delete image
    list_name = retrieveName()[2]
    if username in list_name:
        db.delete(key=username)
        output["remove status"] = "success"
    else:
        output["remove status"] = "unsuccessful: name doesn't exist"

    return output

# recognition
@app.post("/face-recognition/recognise")
def faceRecognition(upload_file: UploadFile = File()):
    filename = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)) + ".jpg"
    file_path = f"{unk_dir}/{filename}"
    with open(file_path, "wb+") as file_object:
        file_object.write(upload_file.file.read())

    # output
    output = {
        "face detected": False,
        "face recognized" : False,
        "name": None,
        "distance": None,
        "similarity (%)": None,
        "all result": None
    }

    # check whether face detected or not
    # if face detected, face will be processed
    try:
        face_detection = DeepFace.detectFace(img_path = file_path,
                                             target_size = (224, 224),
                                             detector_backend = 'ssd'
                                             )
    except:
        output["face detected"] = False
        output["face recognized"] = False

    else:
        output["face detected"] = True

        image_preprocess = functions.preprocess_face(file_path, target_size = (160, 160), detector_backend='ssd', enforce_detection = False)
        image_target = model.predict(image_preprocess)[0].tolist()
        result = retrieveImage()
        image_target_duplicated = np.array([image_target]*result.shape[0])
        result['target'] = image_target_duplicated.tolist()

        # calculate distance and similarity and store to result_df
        result['distance'] = result.apply(findCosineDistance, axis = 1)
        result['similarity (%)'] = result.apply(findCosineSimilarity, axis = 1)
        result = result.sort_values(by = ['distance']).reset_index(drop = True)
        result = result.drop(columns = ["embedding", "target"])

        # get name, distance, and similarity
        name = result[result['distance'] == min(result['distance'])]['img_name']
        name = name.values[0].split('.')[0]
        distance = min(result['distance'])
        similarity = max(result['similarity (%)'])

        if name is not None:
            # set threshold = 0.3
            if distance <= 0.3:
                output["face recognized"] = True
                output["name"] = name
                output["distance"] = np.round(distance, 2)
                output["similarity (%)"] = np.round(similarity, 2)
                output["all result"] = result.to_dict("list")
            else:
                output["face recognized"] = False
                output["name"] = "unknown"
                output["distance"] = np.round(distance, 2)
                output["similarity (%)"] = np.round(similarity, 2)
                output["all result"] = result.to_dict("list")
        else:
            output["face recognized"] = False
            output["name"] = "unknown"
            output["distance"] = np.round(distance, 2)
            output["similarity (%)"] = np.round(similarity, 2)
            output["all result"] = result.to_dict("list")

    return output

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)