import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_dataset_dir = 'data'
base_dir = 'splitted_data'

# Creating directories for training, validation, and test sets
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Splitting ratio
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Function to get list of files in a directory
def get_files_in_directory(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Process each class directory
for category in os.listdir(original_dataset_dir):
    category_path = os.path.join(original_dataset_dir, category)
    if not os.path.isdir(category_path):
        continue  # Skip any non-directory files

    images = get_files_in_directory(category_path)
    if len(images) == 0:
        continue  # Skip classes with no images

    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio))
    val_images, test_images = train_test_split(temp_images, test_size=test_ratio/(test_ratio + validation_ratio))
    
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
    
    for image in train_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(train_dir, category, image))
        
    for image in val_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(validation_dir, category, image))
        
    for image in test_images:
        shutil.copy(os.path.join(category_path, image), os.path.join(test_dir, category, image))

print("Data splitting completed successfully.")
