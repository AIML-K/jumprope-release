import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    """
    Args:
        `jump_size` (int): Window length from data stream.
    """

    def __init__(self, jump_size:int=4, dropout:float=0.1):
        super().__init__()
        
        self.sizes = jump_size * np.array([1,2,2,1]) # 4,8,8,4
        layers = []
        
        for i in range(len(self.sizes)-1):
            layers.append(nn.BatchNorm1d(self.sizes[i], eps=1e-5, affine=True))
            layers.append(nn.Linear(self.sizes[i], self.sizes[i+1]))
            layers.append(nn.ReLU()),
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(self.sizes[-1], 3))
        layers.append(nn.Softmax(dim=-1))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__=="__main__":
    JUMP_SIZE = 8

    # dummy input: shape of (B, JUMP_SIZE)
    # - B: for batch size i.e., number of data samples to feed into network at once.
    # - JUMP_SIZE: a size to split a raw stream data into chunk, varying this value requires to train another model.
    input_ = torch.randn(size=(1,JUMP_SIZE))
    print(f"Input: {input_}")

    # instantiate PyTorch model.
    model = MLP(jump_size=JUMP_SIZE)
    model.eval()
    print("Model is built.")
    print(model)
    
    # use cuda if available.
    if torch.cuda.is_available():
        model.cuda()
        input_ = input_.cuda()
        print("CUDA is under use.")

    # output: shape of (B, 3)
    # - B: for batch size i.e., number of data samples to feed into network at once.
    # - A row of output has is a probability distribution for each label -- (single jump, double jump, others)
    output = model(input_)
    pred = output.argmax(dim=1)[0]
    if pred == 0:
        pred = "Single"
    elif pred == 1:
        pred = "Double"
    else:
        pred = "Other"
    print(f"Output: {output}")
    print(f"Probability of provided input being single jump: {output[0, 0]:.4f}")
    print(f"Probability of provided input being single jump: {output[0, 0]:.4f}")
    print(f"Probability of provided input being single jump: {output[0, 0]:.4f}")
    print(f"Model prediction: {pred}")