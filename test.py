import torch
import model


# Load the best saved model.
def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

model_load('PTB.pt')

model.eval()