import pickle

def load_model():
    model_path = "./TrainedModel/model.sav"
    model_obj = open(model_path, "rb")
    model = pickle.load(model_obj)
    return model



def get_label(pred):
    dic = {
        0:"low quality",
        1:"normal quality",
        2:"high quality"
    }
    return dic[pred]


