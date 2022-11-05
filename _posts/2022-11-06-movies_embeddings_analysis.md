---
layout: single
title: "Bernoulli Embeddings: A demonstration using the MovieLens dataset (Part 2)"
date: 2022-11-05
use_math: true
comments: true
[]: #classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "scroll"
author_profile: false
excerpt: This post develops demonstrates multiple use cases for the embedded representations of movies.
---


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yabramuvdi/yabramuvdi.github.io/blob/master/_notebooks/movies_embeddings_analysis.ipynb)

This post continues the demonstration of the Bernoulli embeddings that started [here](https://yabramuvdi.github.io/movies_embeddings_estimation). After having estimated the embeddings for all the movies available in our dataset we can now evaluate their quality. We will do this by looking at the nearest neighbors of some movies and solving a couple of analogy tasks. 

## Setup


```python
# install libraries
!pip install annoy
```

```python
# clone the repository with the estimated embeddings
!git clone https://github.com/yabramuvdi/bernoulli-embeddings.git
```


```python
# get data from MovieLens
!wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
```


```python
# unzip data
!unzip ml-25m.zip
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
```


```python
# define data path
model_path = "./bernoulli-embeddings/results/"
data_path = "./ml-25m/"
```

## Load the embeddings


```python
# load the estimated embeddings
rho = np.load(model_path + "embeddings_final.npy")
print(rho.shape) # (num_movies, emb_dimension)
```

    (6083, 50)



```python
# load the dictionary mapping from movies to their position in 
# the embeddings matrix
with open(model_path + "item2idx.pkl", 'rb') as f:
    item2idx = pickle.load(f)

# reverse the dictionary
idx2item = {v:k for k,v in item2idx.items()}

# estimated embeddings have one more row because of the padding
print(len(item2idx))    
```

    6082


## Load movies data


```python
# read the data with the information for movies
df_movies = pd.read_csv(data_path + "movies.csv")
df_movies.columns = ["movie_id", "title", "genres"]

# select only the movies for which we have embeddings
df_movies = df_movies.loc[df_movies["movie_id"].isin(list(item2idx.keys()))]
df_movies
```





  <div id="df-99935ca5-2f67-42c0-8370-760509875308">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59794</th>
      <td>201646</td>
      <td>Midsommar (2019)</td>
      <td>Drama|Horror|Mystery</td>
    </tr>
    <tr>
      <th>59844</th>
      <td>201773</td>
      <td>Spider-Man: Far from Home (2019)</td>
      <td>Action|Adventure|Sci-Fi</td>
    </tr>
    <tr>
      <th>60090</th>
      <td>202429</td>
      <td>Once Upon a Time in Hollywood (2019)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>60095</th>
      <td>202439</td>
      <td>Parasite (2019)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>61005</th>
      <td>204698</td>
      <td>Joker (2019)</td>
      <td>Crime|Drama|Thriller</td>
    </tr>
  </tbody>
</table>
<p>6082 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-99935ca5-2f67-42c0-8370-760509875308')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-99935ca5-2f67-42c0-8370-760509875308 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-99935ca5-2f67-42c0-8370-760509875308');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# create mapping from original movie ID to its name
id2movie = {}
movie2id = {}
for i, row in df_movies.iterrows():
    id2movie[row["movie_id"]] = row["title"]
    movie2id[row["title"]] = row["movie_id"]
```


```python
# test dictionaries
id2movie[100], movie2id["City Hall (1996)"]
```




    ('City Hall (1996)', 100)



## Nearest neighbors


```python
# define auxiliary functions
def build_indexer(vectors, num_trees=10):
    """ we will use a version of approximate nearest neighbors
        (ANNOY: https://github.com/spotify/annoy) to build an indexer
        of the embeddings matrix
    """
    
    # angular = cosine
    indexer = AnnoyIndex(vectors.shape[1], 'angular')
    for i, vec in enumerate(vectors):
        # add word embedding to indexer
        indexer.add_item(i, vec)
        
    # build trees for searching
    indexer.build(num_trees)
    
    return indexer

def find_nn(item_name, annoy_indexer, item2idx, idx2item, movie2id, id2movie, n=10):
    """ function to find the nearest neighbors of a given item
    """
    
    # name to index in original database
    item = movie2id[item_name]
    
    # original index to model index
    item_index = item2idx[item]
    
    nearest_indexes, distances =  annoy_indexer.get_nns_by_item(item_index, n+1, include_distances=True)
    nearest_items = [idx2item[i] for i in nearest_indexes[1:] if i > 0]
    
    # get names of movies
    nearest_movies = [id2movie[i] for i in nearest_items]
    
    return nearest_movies, distances

def list_print(x):
    for i, movie in enumerate(x):
        print(f"{i+1}. {movie}")
```


```python
# create an indexer for our estimated embeddings
indexer_rho = build_indexer(rho, 20000)
```


```python
movie = "Star Wars: Episode VI - Return of the Jedi (1983)"
N = 10
print(f"{N} RHO nearest neighbors of {movie}:\n")
nn_movies, nn_dists = find_nn(movie, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    10 RHO nearest neighbors of Star Wars: Episode VI - Return of the Jedi (1983):
    
    1. Star Wars: Episode IV - A New Hope (1977)
    2. Star Wars: Episode V - The Empire Strikes Back (1980)
    3. Indiana Jones and the Last Crusade (1989)
    4. Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)
    5. Star Trek II: The Wrath of Khan (1982)
    6. Indiana Jones and the Temple of Doom (1984)
    7. Star Trek: First Contact (1996)
    8. Star Wars: Episode I - The Phantom Menace (1999)
    9. Aliens (1986)
    10. Terminator, The (1984)



```python
movie = "Harry Potter and the Order of the Phoenix (2007)"
N = 10
print(f"{N} RHO nearest neighbors of {movie}:\n")
nn_movies, nn_dists = find_nn(movie, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    10 RHO nearest neighbors of Harry Potter and the Order of the Phoenix (2007):
    
    1. Harry Potter and the Half-Blood Prince (2009)
    2. Harry Potter and the Deathly Hallows: Part 1 (2010)
    3. Harry Potter and the Goblet of Fire (2005)
    4. Harry Potter and the Deathly Hallows: Part 2 (2011)
    5. Harry Potter and the Prisoner of Azkaban (2004)
    6. Harry Potter and the Chamber of Secrets (2002)
    7. Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
    8. Pirates of the Caribbean: Dead Man's Chest (2006)
    9. Pirates of the Caribbean: At World's End (2007)
    10. Chronicles of Narnia: The Lion, the Witch and the Wardrobe, The (2005)



```python
movie = "Die Hard (1988)"
N = 10
print(f"{N} RHO nearest neighbors of {movie}:\n")
nn_movies, nn_dists = find_nn(movie, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    10 RHO nearest neighbors of Die Hard (1988):
    
    1. Lethal Weapon (1987)
    2. Indiana Jones and the Last Crusade (1989)
    3. Terminator, The (1984)
    4. Indiana Jones and the Temple of Doom (1984)
    5. Untouchables, The (1987)
    6. Aliens (1986)
    7. Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)
    8. Terminator 2: Judgment Day (1991)
    9. Die Hard 2 (1990)
    10. Hunt for Red October, The (1990)



```python
movie = "Kill Bill: Vol. 1 (2003)"
N = 10
print(f"{N} RHO nearest neighbors of {movie}:\n")
nn_movies, nn_dists = find_nn(movie, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    10 RHO nearest neighbors of Kill Bill: Vol. 1 (2003):
    
    1. Kill Bill: Vol. 2 (2004)
    2. Sin City (2005)
    3. Donnie Darko (2001)
    4. Shaun of the Dead (2004)
    5. Grindhouse (2007)
    6. Snatch (2000)
    7. V for Vendetta (2006)
    8. Battle Royale (Batoru rowaiaru) (2000)
    9. Reservoir Dogs (1992)
    10. Team America: World Police (2004)



```python
movie = "Lion King, The (1994)"
N = 10
print(f"{N} RHO nearest neighbors of {movie}:\n")
nn_movies, nn_dists = find_nn(movie, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    10 RHO nearest neighbors of Lion King, The (1994):
    
    1. Aladdin (1992)
    2. Beauty and the Beast (1991)
    3. Pocahontas (1995)
    4. Mrs. Doubtfire (1993)
    5. Jumanji (1995)
    6. Snow White and the Seven Dwarfs (1937)
    7. Home Alone (1990)
    8. Little Mermaid, The (1989)
    9. Pinocchio (1940)
    10. Hunchback of Notre Dame, The (1996)



```python
movie = "2001: A Space Odyssey (1968)"
N = 10
print(f"{N} RHO nearest neighbors of {movie}:\n")
nn_movies, nn_dists = find_nn(movie, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    10 RHO nearest neighbors of 2001: A Space Odyssey (1968):
    
    1. Blade Runner (1982)
    2. Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
    3. Clockwork Orange, A (1971)
    4. Brazil (1985)
    5. Apocalypse Now (1979)
    6. Alien (1979)
    7. Metropolis (1927)
    8. Barry Lyndon (1975)
    9. Planet of the Apes (1968)
    10. Lawrence of Arabia (1962)



```python
movie = "Exorcist, The (1973)"
N = 10
print(f"{N} RHO nearest neighbors of {movie}:\n")
nn_movies, nn_dists = find_nn(movie, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    10 RHO nearest neighbors of Exorcist, The (1973):
    
    1. Halloween (1978)
    2. Poltergeist (1982)
    3. Carrie (1976)
    4. Omen, The (1976)
    5. Rosemary's Baby (1968)
    6. Jaws (1975)
    7. Misery (1990)
    8. Nightmare on Elm Street, A (1984)
    9. Shining, The (1980)
    10. Texas Chainsaw Massacre, The (1974)


## Analogies

A very interesting, and surprising, use of word embeddings is to find word analogies. The famous example used by [Mikolov et al. (2013)](https://arxiv.org/pdf/1301.3781.pdf) searches for a word $X$ in the embedded space that is similar to "woman" in the same sense that "king" is similar to "man". This task can be expressed in terms of a simple vector arithmetic problem as follows:

$$
\vec{King}^{\,} - \vec{Man}^{\,} = \vec{X}^{\,} - \vec{Woman}^{\,} \\
\vec{King}^{\,} - \vec{Man}^{\,} + \vec{Woman}^{\,} = \vec{X}^{\,}
$$

Mikolov et al. (2013) find that when performing this operation on their trained embeddings, they are able to recover the word "queen".

$$ \vec{King}^{\,} - \vec{Man}^{\,} + \vec{Woman}^{\,} \approx \vec{Queen}^{\,} $$

We will play with this idea and try to extend it to our own domain (i.e. movies. Some of the analogies that we will try to solve are: 


$$ \vec{Star Wars V}^{\,} - \vec{Star Wars IV}^{\,} + \vec{LoR I}^{\,} \approx $$

$$ \vec{Harry Potter 5}^{\,} - \vec{Harry Potter 4}^{\,} + \vec{Kill Bill 1}^{\,} \approx $$


```python
def find_nn_vector(vector, annoy_indexer, item2idx, idx2item, movie2id, id2movie, n=10):
    """ function to find the nearest neighbors of a given vector
    """
    
    # find the nearest neighbor of our query vector
    nearest_indexes, distances =  annoy_indexer.get_nns_by_vector(query_emb,  n+1, include_distances=True)
    nearest_items = [idx2item[i] for i in nearest_indexes[1:] if i > 0]
    
    # get names of movies
    nearest_movies = [id2movie[i] for i in nearest_items]
    
    return nearest_movies, distances
```


```python
# define the movies for the analogy task
movie_pos_1 = "Harry Potter and the Half-Blood Prince (2009)"
movie_neg_1 = "Harry Potter and the Order of the Phoenix (2007)"
movie_pos_2 = "Kill Bill: Vol. 1 (2003)"

# get the embedded representation of our movies of interest
emb_pos_1 = rho[item2idx[movie2id[movie_pos_1]]]
emb_neg_1 = rho[item2idx[movie2id[movie_neg_1]]]
emb_pos_2 = rho[item2idx[movie2id[movie_pos_2]]]

# vector arithmetic
query_emb = emb_pos_1 - emb_neg_1 + emb_pos_2
query_emb.shape
```




    (50,)




```python
print(f"Which movie is similar to: {movie_pos_2} in the same sense that {movie_pos_1} is similar to {movie_neg_1}\n")
N = 10
nn_movies, nn_dists = find_nn_vector(query_emb, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    Which movie is similar to: Kill Bill: Vol. 1 (2003) in the same sense that Harry Potter and the Half-Blood Prince (2009) is similar to Harry Potter and the Order of the Phoenix (2007)
    
    1. Kill Bill: Vol. 2 (2004)
    2. Sin City (2005)
    3. Snatch (2000)
    4. Old Boy (2003)
    5. Memento (2000)
    6. Reservoir Dogs (1992)
    7. V for Vendetta (2006)
    8. Shaun of the Dead (2004)
    9. Battle Royale (Batoru rowaiaru) (2000)
    10. Donnie Darko (2001)



```python
# define the movies for the analogy task
movie_pos_1 = "Star Wars: Episode V - The Empire Strikes Back (1980)"
movie_neg_1 = "Star Wars: Episode IV - A New Hope (1977)"
movie_pos_2 = "Lord of the Rings: The Fellowship of the Ring, The (2001)"

# get the embedded representation of our movies of interest
emb_pos_1 = rho[item2idx[movie2id[movie_pos_1]]]
emb_neg_1 = rho[item2idx[movie2id[movie_neg_1]]]
emb_pos_2 = rho[item2idx[movie2id[movie_pos_2]]]

# vector arithmetic
query_emb = emb_pos_1 - emb_neg_1 + emb_pos_2
query_emb.shape
```




    (50,)




```python
print(f"Which movie is similar to: {movie_pos_2} in the same sense that {movie_pos_1} is similar to {movie_neg_1}\n")
N = 10
nn_movies, nn_dists = find_nn_vector(query_emb, indexer_rho, item2idx, idx2item, movie2id, id2movie, N)
list_print(nn_movies)
```

    Which movie is similar to: Lord of the Rings: The Fellowship of the Ring, The (2001) in the same sense that Star Wars: Episode V - The Empire Strikes Back (1980) is similar to Star Wars: Episode IV - A New Hope (1977)
    
    1. Lord of the Rings: The Two Towers, The (2002)
    2. Lord of the Rings: The Return of the King, The (2003)
    3. Matrix, The (1999)
    4. Pirates of the Caribbean: The Curse of the Black Pearl (2003)
    5. Shrek (2001)
    6. Indiana Jones and the Last Crusade (1989)
    7. Star Wars: Episode V - The Empire Strikes Back (1980)
    8. Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)
    9. Kill Bill: Vol. 1 (2003)
    10. Batman Begins (2005)

