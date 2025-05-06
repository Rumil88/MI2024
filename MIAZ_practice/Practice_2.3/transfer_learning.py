import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
import time
import copy
import os
import matplotlib.pyplot as plt

# 1. Підготовка даних
data_dir = os.path.join(os.path.dirname(__file__))  # Use current directory for dataset

print('\n=== Конфігурація навчання ===\n')
print(f'Директорія даних: {data_dir}')

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'\nРозмір датасету:')
for x in ['train', 'val']:
    print(f'{x}: {dataset_sizes[x]} зображень')
print(f'Класи: {class_names}')
print(f'Пристрій: {device}')

# 2. Завантаження попередньо навченого ResNet
model = models.resnet50(pretrained=True)

# 3. Заморожування шарів
for param in model.parameters():
    param.requires_grad = False

# 4. Заміна вихідного шару
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

# 5. Визначення критерію та оптимізатора
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 6. Навчання моделі
num_epochs = 25
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
since = time.time()

# Списки для збереження метрик
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print('\n=== Початок навчання ===\n')
print(f'Кількість епох: {num_epochs}')
print(f'Розмір батчу: {dataloaders["train"].batch_size}')
print(f'Оптимізатор: {type(optimizer).__name__}')
print(f'Швидкість навчання: {optimizer.param_groups[0]["lr"]}\n')

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)
    
    # Кожен епох розділений на тренувальний та валідаційний
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # режим навчання
        else:
            model.eval()   # режим валідації
        
        running_loss = 0.0
        running_corrects = 0
        
        # Ітерація по даних
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Прямий прохід
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Зворотний прохід + оптимізація тільки в тренувальному режимі
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        print(f'{phase.capitalize():5} - Втрати: {epoch_loss:.4f}, Точність: {epoch_acc:.4f}')
        
        # Збереження метрик
        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())
        else:
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc.item())
        
        # Збереження найкращої моделі
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    print()
    
time_elapsed = time.time() - since

print('\n=== Результати навчання ===\n')
print(f'Загальний час навчання: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Найкраща точність на валідації: {best_acc:.4f}')

# Завантаження найкращої моделі
model.load_state_dict(best_model_wts)

# Збереження моделі
torch.save(model.state_dict(), 'fine_tuned_resnet.pth')
print('\nМодель збережено у файлі: fine_tuned_resnet.pth')

# Візуалізація результатів навчання
plt.figure(figsize=(12, 4))

# Графік втрат
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Втрати при навчанні')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()

# Графік точності
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Точність при навчанні')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.close()

print('\nГрафіки результатів навчання збережено у файлі: training_results.png')