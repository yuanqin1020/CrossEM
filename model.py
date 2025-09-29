import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GraphSage
from transformers import CLIPModel, CLIPTokenizer
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, vertex_data, image_data):
        self.vertex_data = vertex_data
        self.image_data = image_data

    def __len__(self):
        return len(self.vertex_data)

    def __getitem__(self, idx):
        vertex = self.vertex_data[idx]
        image = self.image_data[idx]
        return vertex, image

graph_sage = GraphSage(input_dim, hidden_dim, num_layers)
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

vertex_data = [...]  
image_data = [...]  

dataset = CustomDataset(vertex_data, image_data)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

loss_fn = nn.CosineEmbeddingLoss()

optimizer = optim.Adam(graph_sage.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_vertices, batch_images in dataloader:
        optimizer.zero_grad()

        vertex_representations = graph_sage(batch_vertices)

        image_representations = []
        for image in batch_images:
            image_tokenized = clip_tokenizer(image, return_tensors='pt').input_ids
            image_representation = clip_model.encode_image(image_tokenized)
            image_representations.append(image_representation)

        image_representations = torch.cat(image_representations)

        similarity_scores = torch.cosine_similarity(vertex_representations, image_representations)

        target_similarity_scores = torch.ones_like(similarity_scores)

        loss = loss_fn(similarity_scores, target_similarity_scores)

        loss.backward()
        optimizer.step()
        

def inference(dataloader):
    with torch.no_grad():
        for batch_vertices, batch_images in dataloader:
            vertex_representations = graph_sage(batch_vertices)

            image_representations = []
            for image in batch_images:
                image_tokenized = clip_tokenizer(image, return_tensors='pt').input_ids
                image_representation = clip_model.encode_image(image_tokenized)
                image_representations.append(image_representation)

            image_representations = torch.cat(image_representations)

            similarity_scores = torch.cosine_similarity(vertex_representations, image_representations)

            matching_pairs = torch.argmax(similarity_scores, dim=1)

            print(matching_pairs)
