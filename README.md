# Deep Learning Assignment — Fruit Image Classification

This repository contains a Jupyter Notebook (`DLAssgn.ipynb`) which trains a simple convolutional neural network (CNN) to classify fruit images. The notebook covers dataset splitting, data augmentation, model definition, training, saving the trained model, and producing evaluation metrics (classification report and confusion matrix).

## Problem statement

Build and train a CNN to classify images of fruits into their respective categories. The goal is to demonstrate an end-to-end image classification pipeline using TensorFlow / Keras: dataset organization, augmentation, model training, and evaluation.

## Dataset used

- The notebook expects an image dataset organized into class subdirectories (one subfolder per fruit class). Example source location used in the notebook: `/content/data/Fruits_1/Training`.
- The notebook uses `split-folders` to split the dataset into training and testing sets with an 80/20 ratio.
- Images are resized to 100×100 and rescaled to the [0,1] range during preprocessing.

Dataset specifics (this project)

- Dataset origin: extracted from the FRUITS 360 dataset.
- Kaggle source: https://www.kaggle.com/datasets/souro12/ccxzvv

- Classes and image counts included in this assignment:
	- Pineapple: 369 images
	- Pineapple Mini: 450 images
	- Raspberry: 420 images
	- Redcurrant: 492 images
	- Strawberry: 450 images
	- Strawberry Wedge: 496 images

If you don't have the dataset locally, prepare a folder structure like:

```
data/
	Fruits_1/
		Training/
			Pineapple/
			Pineapple Mini/
			Raspberry/
			Redcurrant/
			Strawberry/
			Strawberry Wedge/
```

And then call the split routine to create `Fruits_1_split/train` and `Fruits_1_split/test`.

## Model architecture

The model defined in `DLAssgn.ipynb` is a straightforward sequential CNN:

- Input: 100×100 RGB images
- Conv2D(32, 3×3) → ReLU
- MaxPooling2D(2×2)
- Conv2D(64, 3×3) → ReLU
- MaxPooling2D(2×2)
- Conv2D(128, 3×3) → ReLU
- MaxPooling2D(2×2)
- Flatten
- Dense(128) → ReLU
- Dropout(0.3)
- Dense(N_classes) → softmax

The notebook compiles the model with the Adam optimizer, categorical cross-entropy loss, and accuracy as the reported metric.




## How to run this notebook

Open `DLAssgn.ipynb` in Jupyter Notebook / JupyterLab and run cells sequentially from top to bottom. Below are the exact notebook cells and the actions they perform so others can reproduce the workflow without extra, general environment instructions.

1. Cell 1 — Check GPU (optional)

	 - Commands in the cell:

		 !nvidia-smi

		 import tensorflow as tf
		 print(tf.test.gpu_device_name())

	 - Purpose: Verify GPU presence. This cell is safe to run on systems without GPUs — it will simply report none.

2. Cell 2 — Install Keras (run only if needed in your environment)

	 - Command in the cell:

		 !pip install tensorflow.keras

	 - Purpose: Ensure TensorFlow Keras is available inside the notebook runtime. On managed environments (Colab/remote kernels) this may be required; on local setups you can skip if already installed.

3. Cell 3 — Imports

	 - Contains the exact imports used by the notebook (tensorflow, keras layers, ImageDataGenerator, matplotlib, os).

4. Cell 4 — (Colab / Drive users only) Unzip dataset to /content/data

	 - Command in the cell (example used in the notebook):

		 !unzip -q "/content/drive/MyDrive/archive.zip" -d "/content/data"

	 - Purpose: Only run or adapt this cell if your dataset is stored in Google Drive. If your dataset is local, do not run this cell — instead ensure the local dataset path is set in Cell 6.

5. Cell 5 — Install and import split-folders

	 - Commands in the cell:

		 !pip install split-folders
		 import splitfolders

6. Cell 6 — Set dataset input/output folder variables

	 - Example variables used in the notebook; update these to match your local paths if needed:

		 input_folder = "/content/data/Fruits_1/Training"
		 output_folder = "/content/data/Fruits_1_split"

	 - For this project (local repo layout) use the repository-relative paths if you placed the dataset in the repo root:

		 input_folder = "data/Fruits_1/Training"
		 output_folder = "data/Fruits_1_split"

	 - Note: On Windows specify backslashes or raw strings if editing these variables in a code cell (e.g. r"data\\Fruits_1\\Training").

7. Cell 7 — Split dataset into train/test (80/20)

	 - Exact command in the notebook:

		 splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.2))

	 - Purpose: Creates `data/Fruits_1_split/train` and `data/Fruits_1_split/test` with class subfolders.

8. Cell 8 — Create ImageDataGenerator instances and prepare generators

	 - Key parameters used by the notebook:

		 - target_size=(100, 100)
		 - batch_size=32
		 - class_mode='categorical'

	 - Ensure `train_dir` and `test_dir` in the cell point to:

		 train_dir = 'data/Fruits_1_split/train'
		 test_dir = 'data/Fruits_1_split/test'

9. Cell 9 — Define and compile the model

	 - Model used in the notebook (Sequential with three Conv2D+MaxPool blocks, Flatten, Dense(128), Dropout(0.3), Dense(6, softmax)).
	 - The notebook compiles with:

		 optimizer='adam'
		 loss='categorical_crossentropy'
		 metrics=['accuracy']

	 - Important: Keep `Dense(6, activation='softmax')` only if you are using the six classes listed in this README. If you change classes, update the final Dense layer units accordingly.

10. Cell 10 — Train the model

		- Training command in the notebook:

			with tf.device('/GPU:0'):
					history = model.fit(
							train_gen,
							epochs=20,
							validation_data=test_gen
					)

		- On machines without a GPU, remove the `with tf.device('/GPU:0'):` wrapper or run as-is (TensorFlow will fall back to CPU).

11. Cell 11 — Plot training/validation accuracy

		- The cell reads `history.history['accuracy']` and `history.history['val_accuracy']` and produces the accuracy plot.

12. Cell 12 — Save trained model

		- Command in the notebook:

			model.save('/content/drive/MyDrive/fruit_classifier_cnn.h5')

		- Edit this path if you want the model saved in the repository (for example: `model.save('models/fruit_classifier_cnn.h5')`).

13. Cell 13 — Evaluation and classification report

		- Exact evaluation steps used in the notebook:

			test_gen.reset()
			Y_true = test_gen.classes
			class_indices = test_gen.class_indices
			class_names = list(class_indices.keys())
			Y_pred_probs = model.predict(test_gen)
			Y_pred = np.argmax(Y_pred_probs, axis=1)
			print(classification_report(Y_true, Y_pred, target_names=class_names))

Notes and path adjustments

- The notebook was developed with a set of `/content`-style paths (Google Colab). If you run locally, change `/content/...` paths to your local repository paths (for example `data/Fruits_1/Training` and `data/Fruits_1_split`).
- The notebook expects six classes (Pineapple, Pineapple Mini, Raspberry, Redcurrant, Strawberry, Strawberry Wedge). Leave the final Dense layer at 6 units unless you change the class set.
- Run the cells in the order shown. If you see cells that install packages (pip installs), run them only if the packages are missing in your kernel.

## Evaluation metrics and results

The notebook computes and saves the following metrics:

- Training/validation accuracy per epoch (stored in `history.history['accuracy']` and `history.history['val_accuracy']`). Use the provided plotting cell to visualize learning curves.
- Final classification report (precision, recall, f1-score) using scikit-learn's `classification_report`:

```python
from sklearn.metrics import classification_report, confusion_matrix
Y_true = test_gen.classes
Y_pred_probs = model.predict(test_gen)
Y_pred = np.argmax(Y_pred_probs, axis=1)
print(classification_report(Y_true, Y_pred, target_names=class_names))
```

- Confusion matrix via `confusion_matrix(Y_true, Y_pred)` to inspect per-class errors.





## Files

- `DLAssgn.ipynb` — main notebook with data prep, model, training, and evaluation.
- `README.md` — this file.

