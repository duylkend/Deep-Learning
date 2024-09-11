import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Define the CNN architecture using VGG-16
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()

        # Load the pre-trained VGG-16 model
        vgg16 = models.vgg16(pretrained=True)

        # Freeze VGG-16's feature extractor (optional, if you don't want to fine-tune the convolutional layers)
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # Use VGG-16's convolutional layers as the feature extractor
        self.features = vgg16.features

        # Modify the classifier to match the output dimensions (custom fully connected layers)
        # We replace the original VGG-16 classifier with custom layers
        self.fc1 = nn.Linear(512 * 7 * 7, 512)  # Assuming input images are 224x224, so the output after VGG-16 conv layers is 512x7x7
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input through the VGG-16 feature extractor
        x = self.features(x)
        
        # Flatten the output from the feature extractor
        x = torch.flatten(x, 1)

        # Pass through the custom fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
