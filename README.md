# Movie Recommender with SingleStore and Sentence Transformers

This project demonstrates how to build a simple movie recommendation system using SingleStoreDB as the backend, along with sentence-transformers for embedding movie titles and genres. By creating a table for movies and storing vector embeddings of their combined title-genre-text, you can perform semantic search to find relevant titles based on a given query (for example, "I want to see a sci-fi thriller").

## Prerequisites

1. Python 3.7 or higher.  
2. SingleStoreDB (or a SingleStore Managed Service instance) accessible via a valid connection URL.  
3. Packages:
   - [sentence-transformers](https://pypi.org/project/sentence-transformers/)  
   - [singlestoredb](https://pypi.org/project/singlestoredb/)  
   - [sqlalchemy](https://pypi.org/project/SQLAlchemy/)  
   - [pandas](https://pypi.org/project/pandas/)  

## Project Steps

1. **Install Dependencies:**

   ```bash
   pip install sentence-transformers singlestoredb sqlalchemy pandas sqlalchemy_singlestoredb pymysql
   ```

2. **Database Connection Setup:**

   Replace the placeholder URL with your SingleStore connection URL. In the example script, it is in the format:
   ```
   singlestoredb://USERNAME:PASSWORD@HOST:PORT/DATABASE_NAME
   ```

3. **Database and Table Creation:**

   The script performs the following:
   - Drops the existing `movie_recommender` database if one exists, then creates a fresh database.
   - Creates `movies`, `ratings`, `tags` tables for storing MovieLens data.
   - Fetches the MovieLens dataset (ml-25m) from the GroupLens website and unzips it locally.
   - Loads the CSV files and inserts their contents into the respective tables in SingleStore via `pandas.to_sql`.

4. **Creating a Vector Extended Table:**

   A new table `movie_with_tags_with_vectors` is created to store:
   - `movieId`, `title`, `genres`, `allTags`, and `vector` (the embedding).
   - The `vector` column is a BLOB for storing the embedding from the SentenceTransformer model.

5. **Generating Embeddings:**

   - A SentenceTransformer model (e.g., "flax-sentence-embeddings/all_datasets_v3_mpnet-base") is downloaded.
   - Each movie’s title and genre text are combined, then encoded into a vector using the model.
   - These vectors are inserted into the `movie_with_tags_with_vectors` table.

6. **Retrieving Recommendations:**

   - A sample user search query (e.g., "I want to see a sci-fi thriller") is also converted into an embedding using the same model.
   - SingleStore’s `DOT_PRODUCT` function is used to compute a similarity score between each movie embedding in `movie_with_tags_with_vectors` and the user’s query embedding.
   - Results are ordered by the similarity score in descending order, returning the top N matches.

## Example Usage

Below is a high-level snippet showing how you might connect and run the recommendation query (simplified from the main script):

```python
import sqlalchemy as sa
from singlestoredb import create_engine
from sentence_transformers import SentenceTransformer

# Example SingleStore connection URL:
SINGLESTORE_CONNECTION_URL = "singlestoredb://USERNAME:PASSWORD@HOST:PORT/movie_recommender"
engine = create_engine(SINGLESTORE_CONNECTION_URL)
conn = engine.connect()

# SentenceTransformer model
model = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")

search_query = "I want to see a sci-fi thriller"
search_embedding = model.encode(search_query)

sql_query = sa.text("""
    SELECT 
      title, 
      genres, 
      DOT_PRODUCT(vector, :vector) AS score
    FROM movie_with_tags_with_vectors
    ORDER BY score DESC
    LIMIT 10
""")

results = conn.execute(sql_query, {"vector": search_embedding})
for res in results:
    print(f"{res.title} - {res.genres} (Score: {res.score})")
```

## Notes

- Ensure your SingleStore instance is properly configured to store and process vector data.  
- The example uses `DOT_PRODUCT` for a similarity metric, but SingleStore also supports other functions like `L2_DISTANCE` depending on your use case.  
- If you encounter data-type or chunk-size issues when inserting large datasets, adjust your chunk size or column types accordingly.

## License and Acknowledgments

- This project uses the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/) made available by [GroupLens](https://grouplens.org/).  
- The example script relies on open-source libraries mentioned above. Check their respective licenses for details.  

