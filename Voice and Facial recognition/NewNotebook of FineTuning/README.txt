The pipeline is configured for:
C:\Users\Shahzaib Hussain\Desktop\Shahzaib Hussain\FYP\Datasets\Facial Dataset\fer2013plus\fer2013

The dataset currently contains official train/test folders and 8 classes. The notebook creates validation only from the official training folder.

The notebook supports:
- image_folder/class_name/image files
- CSV files with path and label columns
- FER2013 CSV files with pixels, emotion, and optional Usage columns

All three existing .pt checkpoints are discovered automatically. Set model.name to all to fine-tune EfficientNet-B2, ResNet50, and ViT-B/16, then create probability-level fusion.

The notebook reports training/validation accuracy, loss, macro-F1, precision, recall, test metrics, ROC curves, and row-normalized percentage confusion matrices. It saves and reloads the final reusable fused model as outputs/fused_model.pt; temporary component checkpoints are removed after verification.
