import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import STP_Model
from loader import MoleculeDataset
from dataloader import DataLoaderMaskingPred
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def main():

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset("./dataset")
    
    X_train, X_test = train_test_split(dataset, test_size=0.1, random_state=42)
    train_loader = DataLoaderMaskingPred(X_train, batch_size=256, shuffle = False, num_workers= 4, mask_rate=0.3)
    val_loader = DataLoaderMaskingPred(X_test, batch_size=256, shuffle = False, num_workers= 4, mask_rate=0.3)

    model = STP_Model(in_dim= 58, edge_in_dim=10)
    model.to(device)
    optimizer_model = optim.Adam(model.parameters(), lr=0.001)
    loss_funct = nn.CrossEntropyLoss()

    list_train_loss = []
    list_val_loss = []

    loss_check = 100
    for epoch in range(50):
        model.train()
        train_loss= 0 
        val_loss = 0
        for _, batch in enumerate(train_loader):
            batch = batch.to(device)
            node_rep = model(batch)
            optimizer_model.zero_grad()
            inter_loss = loss_funct(node_rep, batch.mask_node_label.view(-1))
            inter_loss.backward()
            optimizer_model.step()
            train_loss += inter_loss.item()

        model.eval()
        for _, batch in enumerate(val_loader): 
            batch = batch.to(device)
            node_rep = model(batch)  
            mask_node_label = batch.mask_node_label
            inter_loss = loss_funct(node_rep, mask_node_label.view(-1))
            val_loss += inter_loss.item()

        print(f'Epoch {epoch}: ', train_loss/len(train_loader))
        print(f'Epoch {epoch}: ', val_loss/len(val_loader))

        list_train_loss.append(train_loss/len(train_loader))
        list_val_loss.append(val_loss/len(val_loader))

        # if val_loss <= loss_check:
        #     loss_check = val_loss
        torch.save(model.state_dict(), f'./checkpoint/model_{epoch}.pth')
    

if __name__ == "__main__":
    main()
