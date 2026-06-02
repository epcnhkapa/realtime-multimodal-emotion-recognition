# Training Notebooks

Google Colab notebooks used to train the three models in this project.
Each notebook is self-contained: data download, preprocessing, model
definition, training, evaluation, and export of the final `.keras` file
to Google Drive.

## Files

| Notebook | Trained model | Test accuracy |
|---|---|---|
| `vision_training.ipynb` | `vision_ferplus.keras` | ~79.4% (6 classes) |
| `ser_training.ipynb`    | `ser_cnn_bilstm.keras` | ~69.4% (6 classes) |
| `nlp_training.ipynb`    | `mixed_nlp.keras` | ~72% (3 classes) |

## How to run

1. Upload the notebook to Google Colab
2. Mount Google Drive when prompted
3. Run all cells

Datasets are downloaded inside each notebook. RAVDESS and CREMA-D for
SER, FER2013 + FER+ labels for Vision, and a mixed English/Turkish
sentiment dataset for NLP.

## Notes

The exported `.keras` files are the ones bundled with the desktop
application under `application/models/`. Re-running a notebook will
produce a model with similar but not identical weights due to random
initialization, augmentation randomness, and (in Vision) mixup sampling.
