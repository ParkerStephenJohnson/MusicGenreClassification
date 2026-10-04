# Music Genre Classification

Classifying 1,000 songs into 10 genres from features extracted from the audio, comparing three scikit-learn models. Final project for a machine learning course at Santa Clara University, December 2019.

## Data

The [GTZAN genre collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification): 1,000 thirty-second clips, 100 for each of blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae and rock.

I extracted 28 features per clip with [Librosa](https://librosa.org/) and grouped them into two sets:

| Set | Features | File |
|---|---|---|
| Beat and spectral | Tempo, beats, chroma, RMS energy, spectral centroid, bandwidth, roll-off, zero-crossing rate (8) | `data_others.csv` |
| MFCCs | Mel-frequency cepstral coefficients 1 to 20 | `data_mfccs.csv` |
| Both | All 28 | `data_all.csv` |

## Models

- Logistic regression (multinomial, softmax)
- One-vs-rest linear SVM
- Random forest (1,000 trees)

Features are standardized, then split 67/33 into train and test sets.

## Results

Test accuracy by feature set. Random guessing scores about 0.10.

| Features | Logistic regression | One-vs-rest SVM | Random forest |
|---|---|---|---|
| Beat and spectral (8) | 0.43 | 0.47 | 0.49 |
| MFCCs (20) | 0.54 | 0.50 | 0.57 |
| All (28) | 0.61 | 0.58 | 0.61 |

What I found:

- Combining both feature sets beat either set alone for every model.
- Classical, metal and pop were the easiest genres to pick out. Rock was the hardest, with predictions spread across four other genres.
- The random forest reached 1.0 training accuracy on every feature set, so it overfit with only 100 clips per genre.

The full write-up is in [Music Genre Classification.pdf](Music%20Genre%20Classification.pdf).

## Running it

The notebook `main.ipynb` was written on Kaggle and reads the CSVs from Kaggle input paths. To run it locally, point the three `pd.read_csv` calls in the first cell at the CSV files in this repo.

```bash
pip install scikit-learn pandas numpy seaborn matplotlib jupyter
jupyter notebook main.ipynb
```
