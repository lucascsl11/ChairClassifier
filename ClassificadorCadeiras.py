"""
    Trabalho Final - LAMIA
    Esse projeto para a vigésima oitava entrega do Bootcamp de Machine Learning consiste em um problema de classificação,
    utilizando o modelo ResNet-50, que diferencia e classifica diferentes imagens de cadeiras, poltronas e etc.
"""

"""================ Importações ================"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import numpy as np
import zipfile

"""================ Importações ================"""
zip_file_path = 'dataset.zip'
extract_path = './dataset'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"{zip_file_path} extraído para {extract_path}")

"""================ Definições ================"""
data_dir = 'dataset'

# Número alvo de imagens para treino e teste. Mudei do 75/25 para 60/40
target_train_images = 60 
target_val_images = 40

# Transformações para cada classe de imagem. São feitas inicialmente antes das épocas de treinamento.
class_transforms = {
    'Banco de Parque': transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Banquinho': transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Cadeira': transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Poltrona': transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Espreguiçadeira': transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Sofá': transforms.Compose([
        transforms.RandomResizedCrop(100),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, shear=5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Função pra salvar as imagens modificadas e as sintéticas também
def save_limited_augmented_images(img, save_dir, class_name, num_images):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_images):
        augmented_img = class_transforms[class_name](img)
        augmented_img = transforms.ToPILImage()(augmented_img)
        augmented_img.save(os.path.join(save_dir, f"{class_name}_{i}.jpg"))

# Função pra equilibrar o dataset criando novas imagens
def balance_dataset(phase, target_images_per_class):
    dataset_dir = os.path.join(data_dir, phase)
    dataset = datasets.ImageFolder(dataset_dir)

    for class_name in dataset.classes:
        class_dir = os.path.join(dataset_dir, class_name)
        class_samples = [s for s in dataset.samples if dataset.classes[s[1]] == class_name]
        num_existing_images = len(class_samples)
        num_augmented_images = target_images_per_class - num_existing_images

        if num_augmented_images > 0:
            img_paths = [s[0] for s in class_samples]
            img_count = 0
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                save_limited_augmented_images(img, class_dir, class_name, num_augmented_images)
                img_count += num_augmented_images
                if img_count >= num_augmented_images:
                    break

# Isso aqui, na realidade, só adiciona imagens pra classe que tiver faltando (espreguiçadeira)
# mas pode ser usado ao modificar os valores alvo para ter mais imagens ainda.
balance_dataset('train', target_train_images)
balance_dataset('val', target_val_images)

"""================ Carregamento ================"""
# São feitas transformações nos dataloaders também, para que haja variedade em cada época.
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transforms.Compose([
    transforms.RandomResizedCrop(100),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transforms.Compose([
    transforms.RandomResizedCrop(100),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

"""================ Modelo ================"""
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6) # Modificando a camada final para as 6 classes do problema

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

"""================ Treinamento ================"""
num_epochs = 10
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        history[f'{phase}_loss'].append(epoch_loss)
        history[f'{phase}_acc'].append(epoch_acc)

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

"""================ Visualização ================"""
def plot_training_curves(history):
    epochs = range(1, len(history['train_acc']) + 1)

    # Plotando accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [h.cpu().numpy() for h in history['train_acc']], label='Train Accuracy')
    plt.plot(epochs, [h.cpu().numpy() for h in history['val_acc']], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    # Plotando loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

plot_training_curves(history)

"""================ Avaliação ================"""
def evaluate_model(model, dataloader, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    # deixar cmap='blues' se não funcionar o roxo que eu coloquei
    sns.heatmap(cm, annot=True, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão')
    plt.show()

    # Resumo
    report = classification_report(y_true, y_pred, target_names=class_names)
    print('Classification Report:\n', report)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

class_names = train_dataset.classes
evaluate_model(model, val_loader, class_names)

"""================ Predição ================"""
def predict_image_class(image_path, model, transform):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Adicionando dimensão de batch para a imagem

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

image_path = 'image_path.jpg'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
predicted_class_index = predict_image_class(image_path, model, transform)
predicted_class_name = class_names[predicted_class_index]
print(f'Classe da imagem prevista: {predicted_class_name}')

