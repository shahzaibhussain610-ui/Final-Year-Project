# FER2013 Deep Attention Network Notebook Summary Report

## Overview

This notebook focuses on the FER2013 dataset as a preparation step for a deep attention-based facial emotion recognition pipeline. The analysis shown in the notebook is a dataset audit and class-balance check before model training. It examines how the official train/test split is distributed across the seven emotion classes and confirms that the dataset is highly imbalanced.

The notebook computes counts by `split` and `label`, orders labels using a fixed class list, and produces both tabular summaries and visual plots. This is important because class imbalance often affects model learning, especially for minority classes such as `disgust`.

## Dataset structure and labels

The notebook evaluates the FER2013 dataset across these seven emotion classes:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The core code groups the dataset by both `split` and `label`, creates a class distribution table, and then plots:

- a grouped bar chart by emotion and split
- a pie chart showing train/test proportions

This gives a direct view of both class-wise balance and overall split composition.

## Sample counts by class and split

The output confirms the following distribution:

| Label | Test | Train |
|---|---:|---:|
| angry | 958 | 3995 |
| disgust | 111 | 436 |
| fear | 1024 | 4097 |
| happy | 1774 | 7215 |
| neutral | 1233 | 4965 |
| sad | 1247 | 4830 |
| surprise | 831 | 3171 |

This shows that the FER2013 dataset follows a clear long-tailed pattern, with `happy` being the most common class and `disgust` being the least common.

## Key findings

### 1. Strong class imbalance

The notebook explicitly reports the class ratio:

- largest / smallest class ratio: approximately 16.5x
- `happy` has 7,215 train samples, while `disgust` has only 436

This is a major limitation for training, because the model may learn to prioritize dominant classes and underperform on minority emotions.

### 2. Official split is preserved

The distribution table shows that the dataset is split into train and test partitions as expected. The train/test split is not random in a way that destroys the official FER2013 benchmark structure; instead, the notebook keeps the standard class-specific split intact.

### 3. Emotion distribution is not uniform

The class imbalance is not minor. In total:

- `happy` is the dominant class
- `disgust` is severely underrepresented
- other classes such as `angry`, `fear`, `neutral`, and `sad` are moderately represented

This means the notebook is identifying a likely source of bias and poor recall in downstream classification models.

## Visual interpretation

The generated bar chart compares each class across the train and test groups, and the pie chart summarizes the ratio of training to test images. The plots visually confirm that:

- the training set dominates overall sample count
- the test set remains proportional but small for minority classes
- class imbalance is a central issue in FER2013

## Importance for the model pipeline

This notebook is not just a generic plotting step; it establishes the data conditions under which the deep attention model will be trained. The results strongly suggest that:

- standard accuracy may be misleading if the model learns the majority class
- minority-class recall and F1 scores will be critical evaluation metrics
- class weighting, augmentation, or resampling may be necessary to improve performance

This EDA step is therefore essential before model training, because it tells us the dataset is not balanced and training should be designed with this in mind.

## Conclusion

The notebook provides a clear and important summary of the FER2013 dataset distribution. It demonstrates that the dataset is highly imbalanced, with severe underrepresentation of `disgust` and dominance of `happy`. These observations are critical for interpreting any trained emotion-recognition model and suggest that performance may be biased toward the majority class unless the training strategy explicitly addresses class imbalance.

In summary, this notebook serves as a foundational exploratory analysis for a deep attention network pipeline and highlights a key challenge in FER2013: class imbalance can significantly affect model fairness, learning behavior, and evaluation quality.
