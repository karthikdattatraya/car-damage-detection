from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

trained_model = None

class_names = [
    'F_Breakage','F_Crushed', 'F_Normal',
    'R_Breakage', 'R_Crushed', 'R_Normal'
]


class carclassifierresnet(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model():
    global trained_model
    if trained_model is None:
        trained_model = carclassifierresnet(num_classes=6, dropout_rate=0.2)
        trained_model.load_state_dict(
            torch.load("fastapp_server/model/saved_model.pth",
                       map_location=torch.device('cpu'))
        )
        trained_model.eval()
    return trained_model


def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]