from fastapi import FastAPI
import torch
import numpy as np

# ----- Load graph -----
adj = np.load("adjacency_full.npy")

# ----- Model definitions (same as training) -----
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj, dtype=torch.float32))
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = torch.einsum("ij,btjf->btif", self.A, x)
        return self.linear(x)


class STGCN(nn.Module):
    def __init__(self, adj, num_features):
        super().__init__()

        self.temporal1 = nn.Conv2d(num_features, 64, (3,1), padding=(1,0))
        self.graph1 = GraphConv(64,64,adj)
        self.temporal2 = nn.Conv2d(64,64,(3,1),padding=(1,0))

        self.temporal3 = nn.Conv2d(64,64,(3,1),padding=(1,0))
        self.graph2 = GraphConv(64,64,adj)
        self.temporal4 = nn.Conv2d(64,64,(3,1),padding=(1,0))

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64,6)

    def forward(self,x):

        x = x.permute(0,3,1,2)

        x = torch.relu(self.temporal1(x))
        x = self.dropout(x)

        x = x.permute(0,2,3,1)
        x = self.graph1(x)
        x = self.dropout(x)

        x = x.permute(0,3,1,2)
        x = torch.relu(self.temporal2(x))
        x = self.dropout(x)

        x = torch.relu(self.temporal3(x))
        x = self.dropout(x)

        x = x.permute(0,2,3,1)
        x = self.graph2(x)
        x = self.dropout(x)

        x = x.permute(0,3,1,2)
        x = torch.relu(self.temporal4(x))
        x = self.dropout(x)

        x = x[:, :, -1, :]
        x = x.permute(0,2,1)
        x = self.fc(x)
        x = x.permute(0,2,1)

        return x


# ----- Load trained model -----
model = STGCN(adj,6)
model.load_state_dict(torch.load("best_stgcn_model.pth", map_location="cpu"))
model.eval()

# ----- FastAPI app -----
app = FastAPI()

@app.get("/")
def root():
    return {"message":"STGCN Traffic Prediction API"}

@app.get("/predict")
def predict_random():

    # random sample input
    sample = torch.randn(1,12,207,6)

    with torch.no_grad():
        pred = model(sample)

    return {"prediction": pred.numpy().tolist()}
    