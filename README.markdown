# Hybrid Movie Recommender System

This repository contains a hybrid movie recommender system that integrates collaborative filtering and content-based filtering using a deep neural network. Built with Python and TensorFlow, the system leverages the MovieLens dataset to provide personalized movie recommendations by predicting user ratings and relevance scores. It includes innovative features like user history analysis and personalized similar movie recommendations, enhancing both accuracy and interpretability.

## Project Overview

The system combines user-movie interaction data (ratings) with content features (genres, keywords) to address challenges like data sparsity and cold-start problems. It employs a dual-task learning framework to predict:

- **Numerical ratings** (0.5–5.0 scale)
- **Binary relevance scores** (rating &gt; 3.5)

Key features include:

- **User History Analysis**: Displays a user’s rated movies, genres, overviews, and ratings to validate preference alignment.
- **Personalized Similar Movie Recommendations**: Suggests movies similar to a user-specified title, tailored to individual preferences using Jaccard similarity.
- **Negative Sampling**: Incorporates unrated movies to handle implicit feedback, improving model robustness.

The project achieves a Mean Absolute Error (MAE) of 0.39, Precision@10 of 0.40, and Recall@10 of 0.75, demonstrating strong performance in delivering accurate and relevant recommendations.

## Dataset

The system uses the MovieLens dataset, including:

- **Movies Metadata**: \~45,000 movies, reduced to 40,000 after preprocessing (removing duplicates, missing overviews, empty genres).
- **Ratings**: \~100,000 ratings from 700 users, reduced to 80,000 after merging with movie metadata.
- **Keywords**: Filtered to the top 1,000 most frequent terms for dimensionality reduction.

Files required:

- `movies_metadata.csv`
- `ratings_small.csv`
- `keywords.csv`

Place these files in the project directory or update the file paths in `load_data()`.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/hybrid-movie-recommender.git
   cd hybrid-movie-recommender
   ```

2. **Install Dependencies**: Ensure Python 3.8+ is installed, then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with:

   ```
   pandas==2.0.3
   numpy==1.24.3
   tensorflow==2.12.0
   scikit-learn==1.3.0
   ```

3. **Download the Dataset**:

   - Download the MovieLens dataset from Kaggle.
   - Place `movies_metadata.csv`, `ratings_small.csv`, and `keywords.csv` in the project directory.

## Usage

Run the main script to train the model and generate recommendations:

```bash
python recommender.py
```

### Example Commands

- **Display User History**:

  ```python
  history_123 = display_user_history(123, movies, ratings)
  print(history_123)
  ```

- **Predict Rating**:

  ```python
  print(predict_rating(model, 123, 'Toy Story', movies, ratings, user_to_index, movie_to_index, genre_to_index, keyword_to_index, max_genres, max_keywords))
  ```

- **General Recommendations**:

  ```python
  recommendations = recommend_movies(model, 123, movies, user_to_index, movie_to_index, genre_to_index, keyword_to_index, max_genres, max_keywords, top_n=5)
  print(recommendations)
  ```

- **Similar Movie Recommendations**:

  ```python
  similar_recommendations = recommend_similar_movies(model, 123, 'Toy Story', movies, ratings, user_to_index, movie_to_index, genre_to_index, keyword_to_index, max_genres, max_keywords, top_n=5)
  print(similar_recommendations)
  ```

### Output

The script:

- Trains the model and saves it as `final_precision_recall_movie_recommender.h5`.
- Outputs performance metrics (Precision@10, Recall@10).
- Displays user history, predicted ratings, and recommendations as tables.

## Model Architecture

The deep neural network consists of:

- **Inputs**: User ID, movie ID, genre sequence, keyword sequence.
- **Embeddings**:
  - User and movie embeddings: 256 dimensions.
  - Genre and keyword embeddings: 32 dimensions.
- **Processing**:
  - User and movie embeddings are flattened.
  - Genre and keyword embeddings are averaged using `GlobalAveragePooling1D`.
  - Concatenated features (576 dimensions) pass through dense layers (128, 64, 32, 16 units) with ReLU activation, batch normalization, and 30% dropout.
- **Outputs**:
  - Rating prediction (0.5–5.0 scale).
  - Binary relevance score (sigmoid, rating &gt; 3.5).
- **Training**:
  - Optimizer: Adam (learning rate 0.0001).
  - Loss: Mean squared error (rating) + binary cross-entropy (relevance, weight 0.85).
  - Callbacks: Early stopping (patience 3), learning rate reduction (factor 0.5, patience 2).

## Results

The model was evaluated on a test set (20% of data):

- **MAE**: 0.39
- **Precision@10**: 0.40
- **Recall@10**: 0.75

These metrics indicate high rating accuracy and strong recall, with moderate precision, suitable for personalized recommendations on streaming platforms.

## Project Structure

```
hybrid-movie-recommender/
├── recommender.py          # Main script with model and functions
├── movies_metadata.csv     # MovieLens movies metadata
├── ratings_small.csv       # MovieLens ratings
├── keywords.csv            # MovieLens keywords
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Future Improvements

- Incorporate additional metadata (e.g., cast, directors).
- Implement sequential modeling (e.g., RNNs) for dynamic user preferences.
- Add real-time feedback integration for adaptive recommendations.
- Explore advanced metrics like NDCG for deeper evaluation.

## Citation

If you use this project, please cite our paper:

```bibtex
@article{mahrous2025hybrid,
  author = {Yehia Mahrous and Omar Elborollosy and Ziad Abdelhafiz},
  title = {A Hybrid Deep Learning Approach for Personalized Movie Recommendations with Content-Based Features},
  journal = {Proc. IEEE Conf.},
  year = {2025}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- MovieLens Dataset for providing the data.
- TensorFlow for the deep learning framework.
- Egypt University of Informatics for supporting this research.