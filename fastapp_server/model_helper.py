from PIL import  Image
import torch
from torch import nn
from torchvision import models,transforms


trained_model=None
class_names = [
 'F_Breakage', 'F_Crushed', 'F_Normal',
 'R_Breakage', 'R_Crushed', 'R_Normal'
]

class carclassifierresnet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        # Get input features of final layer
        in_features = self.model.fc.in_features
        # Replace fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x= self.model(x)
        return x



def predict(image_path):
    image=Image.open(image_path).convert("RGB") #image data ie binary form into RGB
    transform=transforms.Compose([
        transforms.Resize((224,224)), #model trained on 224*224 ,hence  model expects the size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    image_tensor=transform(image).unsqueeze(0) # this outpuy (3,224,224) ,model trained on batches  (32,3,224,224), convert into (1,3,224,224) unsqueeze adds a new dimension

    global  trained_model

    if trained_model is None:
        trained_model=carclassifierresnet()
        trained_model.load_state_dict(torch.load("model/saved_model.pth")) #loading the trained model parameters
        trained_model.eval()

    with torch.no_grad():
        output=trained_model(image_tensor)
        _,predicted_class=torch.max(output,1)
        return class_names[predicted_class.item()]