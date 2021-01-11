import torch
from vit_pytorch import ViT

#image is 640 by 480

device = 'cuda'
gamma = 0.7

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#how to modify to get scalar (speed) rather than classes?


#todo add data loaders

v = ViT(
    image_size = 640, #max(height, width)
    patch_size = 20, #common factor of both height and width
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
