import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileReader


class TextDataset(Dataset):
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = list(extract_text_by_page_v2(pdf_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def extract_text_by_page_v2(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfFileReader(file)
        for page_number in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_number)
            yield page.extract_text()


def export_as_csv_v3(pdf_path, csv_path, title_array):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    row_length = len(title_array)

    data = []
    for page_text in extract_text_by_page_v2(pdf_path):
        text = page_text.replace('\u25cf', ',')
        words = text.split()
        data.append([words[i:i + row_length] for i in range(0, len(words), row_length)])

    flat_data = [item for sublist in data for item in sublist]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(title_array)
        writer.writerows(flat_data)

)

   
    loss = model.evaluate(test_data, test_labels, verbose=2)
    print(f'Test Loss: {loss}')

    
    history = model.history.history
    sns.lineplot(x=range(1, len(history['loss']) + 1), y=history['loss'], label='Training Loss')
    sns.lineplot(x=range(1, len(history['val_loss']) + 1), y=history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.show()


if __name__ == '__main__':
     
    export_as_csv_v3(pdf_path, csv_path, title_array)

 
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        data = [row for row in csv_reader]

    
    data_numerical = [[len(word) for word in row] for row in data]

   
    input_size = len(title_array)
    output_size = 1  # Assuming a regression task, adjust as needed
    data_tensor = torch.tensor(data_numerical, dtype=torch.float32)
    labels_tensor = torch.tensor([[1.0] for _ in range(len(data_numerical))], dtype=torch.float32)  # Dummy labels

  
    train_neural_network(data_tensor.numpy(), labels_tensor.numpy(), input_size, output_size)
