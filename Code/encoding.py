from sklearn.preprocessing import LabelEncoder
from data_collection import Image_Preprocessing
from torch.utils.data import TensorDataset, DataLoader
import torch
label_encode = LabelEncoder()
obj = Image_Preprocessing()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data , label = obj.load_images('/home/dread/Face Recognition/Data')
encoded_labels = label_encode.fit_transform(label)

label_tensor = torch.tensor(encoded_labels).to(device)

data_set = TensorDataset(data , label_tensor)
train_size = int(0.8 * len(data_set))  # 80% for training
val_size = len(data_set) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(data_set, [train_size, val_size])

# Create DataLoader to feed data into the model
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

