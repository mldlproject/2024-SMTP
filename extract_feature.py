import torch
import numpy as np
from model import STP_Model
from loader import load_data_extracted
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = load_data_extracted(f'C:/Users/NC/Desktop/Others/Project 3/source_code/iBBBP-MGTP/data_preparation/data/featurized/mol2vec.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = STP_Model(in_dim= 58, edge_in_dim=10)
    model.load_state_dict(torch.load('./checkpoint/model_47.pth'))
    model.to(device)
    model.eval()
   
    features = []
    for _, batch in enumerate(dataloader):
        batch = batch.to(device)
        molecules_features = model.extract_features(batch)  
        features.append(molecules_features.detach().cpu().numpy())

    features = np.concatenate(features, axis=0)
    print(features.shape)
    np.save(f'C:/Users/NC/Desktop/Others/Project 3/source_code/iBBBP-MGTP/data_preparation/data/featurized/ours_GIN.npy', features)
    return features

if __name__ == "__main__":
    main()
