from project_name.model import Model
from project_name.data import MyDataset
from rice_cnn_classifier.data import RiceDataset

def train():
    dataset = RiceDataset(data_path="data/processed", split="train", transform=get_transforms("train"))
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
