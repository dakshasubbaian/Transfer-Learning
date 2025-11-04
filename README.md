# Implementation-of-Transfer-Learning
## AIM:
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
To develop a deep learning-based classification model using transfer learning with the VGG-19 architecture to accurately classify images of capacitors into two categories: defected and non-defected. The model will be trained on labeled images, optimized using appropriate loss and optimizer functions, and evaluated using metrics such as confusion matrix, classification report, and performance plots.
<br>
This classification system helps automate the quality control process in manufacturing by detecting defective capacitors based on their images, thereby improving efficiency and reducing manual inspection errors. The project leverages pre-trained VGG-19 weights and adapts the final classification layer to suit the specific dataset.

## DESIGN STEPS:
### STEP 1:
Import the necessary libraries such as PyTorch, Torchvision, and Matplotlib.  


### STEP 2:
Load the dataset and apply preprocessing (resizing, normalization, and augmentation).  


### STEP 3:
Download the pre-trained VGG-19 model from Torchvision models.  


### STEP 4:
Freeze the feature extraction layers of VGG-19.  


### STEP 5:
Modify the final fully connected layer to match the number of dataset classes.  


### STEP 6:
Define the loss function (CrossEntropyLoss) and optimizer (Adam/SGD).  


### STEP 7:
Train the model on the training dataset and validate on the validation set.  


### STEP 8:
Plot Training Loss and Validation Loss vs Iterations.  


### STEP 9:
Evaluate the model on the test dataset.  


### STEP 10:
Generate Confusion Matrix, Classification Report, and test on new sample images.  



## PROGRAM:
```python
# Load a pre-trained VGG19 model
model = models.vgg19(pretrained=True)


# Modify the final fully connected layer to match the dataset classes
in_features=model.classifier[-1].in_features
num_classes = len(train_dataset.classes)
model.classifier[-1] = nn.Linear(in_features, 1)

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train the model
def train_model(model, train_loader, test_loader, num_epochs=50):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # Ensure proper shape for BCEWithLogitsLoss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: DAKSHA SUBBAIAN ")
    print("Register Number:212223230036")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT:
### Training Loss, Validation Loss Vs Iteration Plot

<img width="690" height="264" alt="image" src="https://github.com/user-attachments/assets/d720f7c9-e87b-401f-a66c-b08b899999dd" />
<img width="857" height="674" alt="image" src="https://github.com/user-attachments/assets/0280dcc7-002c-446a-aff4-2437a3525353" />





### Confusion Matrix

<img width="658" height="661" alt="image" src="https://github.com/user-attachments/assets/f23ca86d-15cb-4f85-8982-4c6a96a66872" />




### Classification Report

<img width="554" height="255" alt="image" src="https://github.com/user-attachments/assets/397b3e6f-c38e-443f-a064-ea05dc592021" />





### New Sample Prediction

<img width="433" height="501" alt="image" src="https://github.com/user-attachments/assets/9282ec82-2301-44a7-96ab-be7db2d05d8b" />

<img width="428" height="501" alt="image" src="https://github.com/user-attachments/assets/c199b48e-5d48-4338-9117-dbdb3a3d9431" />



## RESULT:
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
