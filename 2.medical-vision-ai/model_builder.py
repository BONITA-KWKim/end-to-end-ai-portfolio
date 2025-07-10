from model.abmil import MILClassifier
from model.self_attention_mil import MILSelfAttentionClassifier
from model.gmlp import gMLPMILPooling


# def get_model(name: str, **kwargs):
def get_model(params, **kwargs):
    name = params["model_name"]
    name = name.lower()

    if name == "abmil":
        return MILClassifier(num_classes=params["num_classes"])
    elif name == "self-attetion-mil":
        return MILSelfAttentionClassifier(num_classes=params["num_classes"])
    elif name == "gmlp":
        return gMLPMILPooling(num_classes=params["num_classes"])
    else:
        raise ValueError(f"Unknown model name: {name}")
