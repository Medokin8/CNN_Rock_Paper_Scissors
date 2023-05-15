import os
import shutil
import random

# Set the source and destination directories
src_dir = "rock"
train_dir = "train/rock"
val_dir = "validation/rock"
test_dir = "test/rock"

# Create the train, validation, and test directories if they don't exist
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(val_dir):
    os.mkdir(val_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Get a list of all the files in the source directory
files = os.listdir(src_dir)

# Shuffle the files randomly
random.shuffle(files)

# Calculate the split points for the train, validation, and test sets
train_split = int(0.7 * len(files))
val_split = int(0.15 * len(files))
test_split = len(files) - train_split - val_split

# Copy the files to the train directory
for file in files[:train_split]:
    src_path = os.path.join(src_dir, file)
    dest_path = os.path.join(train_dir, file)
    shutil.copy(src_path, dest_path)
    os.remove(src_path)

# Copy the files to the validation directory
for file in files[train_split:train_split+val_split]:
    src_path = os.path.join(src_dir, file)
    dest_path = os.path.join(val_dir, file)
    shutil.copy(src_path, dest_path)
    os.remove(src_path)

# Copy the files to the test directory
for file in files[train_split+val_split:]:
    src_path = os.path.join(src_dir, file)
    dest_path = os.path.join(test_dir, file)
    shutil.copy(src_path, dest_path)
    os.remove(src_path)
