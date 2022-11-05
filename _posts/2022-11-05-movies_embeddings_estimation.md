---
layout: single
title: "Bernoulli Embeddings: A demonstration using the MovieLens dataset"
date: 2022-11-01
use_math: true
comments: true
[]: #classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "scroll"
author_profile: false
excerpt: This post develops an extension to traditional word embedding models in order to produce low-dimensional representations of movies.
---


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yabramuvdi/yabramuvdi.github.io/blob/master/_notebooks/movies_embeddings_estimation.ipynb)

This post builds on the Exponential Family Embeddings framework developed by [Rudolph et al. (2016)](https://arxiv.org/pdf/1608.00778.pdf) in order to estimate dense low-dimensional representations of movies. In essence, we will construct these representations by leveraging the consumption patterns of users.

Under the hood, the code that we will use is written in Python and mainly uses JAX to estimate the embeddings. You can find the repository [here](https://github.com/yabramuvdi/bernoulli-embeddings). You can also explore an additional notebook that uses these embedded representations and a [recommender system](https://movies-embeddings.herokuapp.com/) I built with them.

## Motivation, theory, and model

### Motivation

Word embeddings have proven to be powerful low-dimensional continuous representation of words. Essentially, embeddings capture the intuition that a word can be described by the context in which it appears. Words that occur in similar contexts usually have similar meanings. Embeddings are used in many natural language processsing (NLP) tasks, such as predicting the next three words on your smart phone. [Rudolph et al. (2016)](https://arxiv.org/pdf/1608.00778.pdf) extend word embeddings to other types of high-dimensional data, including neural data and movie recommendation systems. <br><br>


We rely on their methodological development to estimate low-dimensional dense representations of movies. Movies are observations with many characteristics in several dimensions: genre, actors, year of release, etc. We can try to compare movies along these many dimensions. But we can also map these multi-dimensional points to a lower latent spaces. We generate these embedded representations of movies by identifying all the movies that were highly ranked by the same user. The main idea is that there is a lot of information encoded in the fact that the same user ranks highly multiple movies. A user that likes Toy Story and Finding Nemo will probably also rank very highly Coco. <br><br>

In terms of mechanics, the goal is to model each movie conditional on a set of other movies (the "context"). Each movie is associated with two latent vectors: an embedding vector $$\rho$$ and a context vector $$\alpha$$. Both jointly determine the conditional probability that relates each movie to its context. We will now develop a little bit more of the model before showing how to estimate it.

### The model

We want to model the conditional probability that a movie $$i$$ is highly ranked by user $$j$$, given the existing set of movies that user $$j$$ has ranked highly. Let $$S_j$$ be the set of all movies that user $$j$$ ranks highly, and $$S_j^{-i}$$ that set without movie $$i$$. <br><br>

We consider the simplest setting, a Bernoulli embedding, in which movie $$i$$ is part of $$S_j$$ with probability $$p_i$$. Then, $$i$$ follows a Bernoulli distribution with parameter $$p_i$$:

$$
\begin{equation}
    i \in S_j | S_j^{-i} \sim Bernoulli(p_i) \quad \textrm{and} \quad p_i \equiv P \left(i \in S_j | S_j^{-i} \right)
\end{equation}
$$

**The fundamental idea behind embeddings is that we can parametrize this conditional probability using two types of low-dimensional continuous vectors: embeddings and context vectors**. For each movie $$i$$ we will have a $$K$$-dimensional embedding vector $$\rho_i = \left[\rho_{i,1}, \rho_{i,2}, ... , \rho_{i,K} \right]$$ and a $$K$$-dimensional context vector $$\alpha_i = \left[\alpha_{i,1}, \alpha_{i,2}, ... , \alpha_{i,K} \right]$$. The choice of $$K$$ is up to the modeler: higher $$K$$ implies more detailed embeddings, but at the expense of computational cost. <br><br>

There are several classes of models that specify how the $$\rho$$ and $$\alpha$$ vectors exactly interact with each other in order to define the conditional probability of interest. We choose a simple parameterization of the embedding structure: (i) a linear combination of the embeddings vectors to model the conditional probabilities, and (ii) a sigmoid link function to map outcomes to admissible probabilities. <br><br>

First, we use a linear combination of the embeddings vectors to model the conditional probability. Intuitively, we want to have the embedding of movie $$i$$, $$\rho_i$$, to be as similar as possible to the context embeddings $$\alpha$$ of the rest of movies that were ranked highly by user $$j$$. Formally, we combine the embedding $$\rho_i$$ with the context embeddings of the other movies by computing: <br><br>

$$
\begin{equation}
  \rho_i^T \sum_{s \in S_j^{-i}} \alpha_s
\end{equation}
$$

Second, we choose a link function to map outcomes to probabilities. We normalize by the number of movies that are part of the context, $$\frac{1}{\#S_j^{-i}}$$, to account for the fact that each user ranks a different number of movies. Then, we use a **sigmoid link function** $$\sigma(\cdot)$$ to transform the value that we have into a number between 0 and 1 to obtain admissible probabilities. <br><br>

These two ingredients lead to our model for a single conditional probability:

$$
\begin{equation}
    P \left(i \in S_j | S_j^{-i} \right) = f_i \left( \rho_i^T \sum_{s \in S_j^{-i}} \alpha_s \right) = \sigma \left( \frac{1}{\#S_j^{-i}} \; \rho_i^T \sum_{s \in S_j^{-i}} \alpha_s \right) 
\end{equation}
$$

In order to construct a complete probability for the observed data, we also need to model the probability of a movie $$g$$ \emph{not} not being ranked by user $$j$$:

$$
\begin{equation}
\begin{aligned}
Pr \left(g \notin S_j | S_j \right) = 1 - Pr \left(g \in S_j | S_j \right) = 1 - \sigma \left( \frac{1}{\#S_j} \rho_g^T \sum_{s \in S_j} \alpha_s \right)
\end{aligned}
\end{equation}
$$

This term needs to be computed for all possible movies $$g$$  that are not ranked by user $$j$$, which makes the problem extremely computationally expensive. However, following the word embeddings literature, we can randomly sample a subset of $$g$$ movies to compute this term. If the size of the random sample is considerably smaller than the number of suppliers, we can compute this term at a reasonable computational cost. This strategy is known as **negative sampling**. 

Bringing everything together, for a single user we have the following conditional log probability:

$$
\begin{equation}
    \mathcal{L}_j(\rho, \alpha) = \sum_{i \in S_j} \log \sigma \left(\frac{1}{\#S_j^{-i}} \rho_i^T \sum_{s \in S_j^{-i}} \alpha_s \right) +  \sum_{k \in NS} \log \left( 1 - \sigma \left( \frac{1}{\#S_j} \rho_k^T \sum_{s \in S_j} \alpha_s \right) \right)
\end{equation}
$$

Summing over all observations, and including regularization terms on both sets of embeddings, the full objective function is given by

$$
\begin{equation}
   \begin{aligned}
           \mathcal{L}(\rho, \alpha) & = \sum_{j \in C} \sum_{i \in S_j} \log \sigma \left(\frac{1}{\#S_j^{-i}} \rho_i^T \sum_{s \in S_j^{-i}} \alpha_s \right) + \\ & 
           \gamma \sum_{j \in C} \sum_{k \in NS} \log \left( 1 - \sigma \left( \frac{1}{\#S_j^{-i}} \rho_k^T \sum_{s \in S_j} \alpha_s \right) \right)  +
           \log p (\rho)+ \log p (\alpha)
   \end{aligned} 
\end{equation}
$$

The first term describes the conditional probability of a movie $$i$$ being highly ranked by user $$j$$. The second term correspond to the negative samples, and the last two terms describe the priors we have on both embeddings vectors.

## Estimation

We will use our own custom code (available [here](https://github.com/yabramuvdi/bernoulli-embeddings)) in order to estimate these embeddings.

### Setup


```python
# check that we got GPU
!nvidia-smi
```

    Wed Nov  2 17:11:47 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   40C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


```python
# install a optimization library for JAX
!pip install optax
```

```python
# clone the repository with the source code
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
import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import optax

# add path to read our source code
sys.path.insert(0, "./bernoulli-embeddings/src")
import data_generation as dg
import bernoulli_embeddings as be
```


```python
# define data path
input_path = "./ml-25m/"
```


```python
# check if GPU is being used by JAX
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```

    gpu


### MovieLense data

The MovieLens 25M dataset contains the ratings given by 162,000 users to 62,000 movies. We will simplify the data by keeping only the movies that were rated with 5 stars and dropping movies that were rated by very few users. This way we are able to focus only on the set of movies that users clearly liked and were watched by many users.
<br><br>

Check the [official webpage](https://grouplens.org/datasets/movielens/) for more information.



```python
# read original data
df = pd.read_csv(input_path + "ratings.csv")
df
```





  <div id="df-6133dbbd-c297-4e22-9077-28d34796c6b4">
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>296</td>
      <td>5.0</td>
      <td>1147880044</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>306</td>
      <td>3.5</td>
      <td>1147868817</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>307</td>
      <td>5.0</td>
      <td>1147868828</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>665</td>
      <td>5.0</td>
      <td>1147878820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>899</td>
      <td>3.5</td>
      <td>1147868510</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25000090</th>
      <td>162541</td>
      <td>50872</td>
      <td>4.5</td>
      <td>1240953372</td>
    </tr>
    <tr>
      <th>25000091</th>
      <td>162541</td>
      <td>55768</td>
      <td>2.5</td>
      <td>1240951998</td>
    </tr>
    <tr>
      <th>25000092</th>
      <td>162541</td>
      <td>56176</td>
      <td>2.0</td>
      <td>1240950697</td>
    </tr>
    <tr>
      <th>25000093</th>
      <td>162541</td>
      <td>58559</td>
      <td>4.0</td>
      <td>1240953434</td>
    </tr>
    <tr>
      <th>25000094</th>
      <td>162541</td>
      <td>63876</td>
      <td>5.0</td>
      <td>1240952515</td>
    </tr>
  </tbody>
</table>
<p>25000095 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6133dbbd-c297-4e22-9077-28d34796c6b4')"
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
          document.querySelector('#df-6133dbbd-c297-4e22-9077-28d34796c6b4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6133dbbd-c297-4e22-9077-28d34796c6b4');
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
# rename columns
df.columns = ["user_id", "movie_id", "rating", "time"]

# keep only the ratings for movies that people loved (5 stars)
df = df.loc[df["rating"] > 4]
```

We will use the more generic names of *basket* (instead of user) and *item* (instead of movie). In a way, users can be thought of as baskets that contain multiples items (i.e. all the movies that they have liked).


```python
# simplify and rename columns
df = df[["user_id", "movie_id"]]
df.columns = ["basket", "item"]
df
```





  <div id="df-239878b2-6a60-4fac-8812-95adbf50a98e">
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
      <th>basket</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>307</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>665</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1237</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2351</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25000085</th>
      <td>162541</td>
      <td>8983</td>
    </tr>
    <tr>
      <th>25000086</th>
      <td>162541</td>
      <td>31658</td>
    </tr>
    <tr>
      <th>25000089</th>
      <td>162541</td>
      <td>45517</td>
    </tr>
    <tr>
      <th>25000090</th>
      <td>162541</td>
      <td>50872</td>
    </tr>
    <tr>
      <th>25000094</th>
      <td>162541</td>
      <td>63876</td>
    </tr>
  </tbody>
</table>
<p>5813013 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-239878b2-6a60-4fac-8812-95adbf50a98e')"
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
          document.querySelector('#df-239878b2-6a60-4fac-8812-95adbf50a98e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-239878b2-6a60-4fac-8812-95adbf50a98e');
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
# drop movies that appear less than a min number of times
df = dg.drop_items(df, min_times=50)
df
```





  <div id="df-f93ccf27-b5fc-4046-8cab-acf6a0f9ec67">
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
      <th>basket</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>296</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>296</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>296</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5639968</th>
      <td>151652</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5639969</th>
      <td>156667</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5639970</th>
      <td>158748</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5639971</th>
      <td>160329</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5639972</th>
      <td>160543</td>
      <td>1976</td>
    </tr>
  </tbody>
</table>
<p>5639973 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f93ccf27-b5fc-4046-8cab-acf6a0f9ec67')"
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
          document.querySelector('#df-f93ccf27-b5fc-4046-8cab-acf6a0f9ec67 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f93ccf27-b5fc-4046-8cab-acf6a0f9ec67');
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
# remove any basket that has less than 2 items
df = dg.check_baskets(df, min_items=2)
df
```





  <div id="df-801be144-99bb-4dad-86ce-43944bd08cd5">
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
      <th>basket</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>296</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>296</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>296</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>296</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5637129</th>
      <td>151652</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5637130</th>
      <td>156667</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5637131</th>
      <td>158748</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5637132</th>
      <td>160329</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>5637133</th>
      <td>160543</td>
      <td>1976</td>
    </tr>
  </tbody>
</table>
<p>5637134 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-801be144-99bb-4dad-86ce-43944bd08cd5')"
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
          document.querySelector('#df-801be144-99bb-4dad-86ce-43944bd08cd5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-801be144-99bb-4dad-86ce-43944bd08cd5');
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




### Prepare data

In order to estimate our embeddings we need to transform the data structure we currently have. Instead of having pairs of (basket, item) we want to group together all the items that belong to the same basket.


```python
# generate indexes for all the baskets and items in our data
df_train, basket2idx, item2idx = dg.gen_indexes(df, 
                                                basket_col="basket",
                                                item_col="item")
df_train
```





  <div id="df-b03795b1-77d0-4028-b636-1b465ce4cf6d">
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
      <th>basket</th>
      <th>item</th>
      <th>basket_idx</th>
      <th>item_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>296</td>
      <td>1</td>
      <td>226</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>296</td>
      <td>3</td>
      <td>226</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>296</td>
      <td>8</td>
      <td>226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>296</td>
      <td>10</td>
      <td>226</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>296</td>
      <td>12</td>
      <td>226</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5637129</th>
      <td>151652</td>
      <td>1976</td>
      <td>147020</td>
      <td>1237</td>
    </tr>
    <tr>
      <th>5637130</th>
      <td>156667</td>
      <td>1976</td>
      <td>151901</td>
      <td>1237</td>
    </tr>
    <tr>
      <th>5637131</th>
      <td>158748</td>
      <td>1976</td>
      <td>153928</td>
      <td>1237</td>
    </tr>
    <tr>
      <th>5637132</th>
      <td>160329</td>
      <td>1976</td>
      <td>155468</td>
      <td>1237</td>
    </tr>
    <tr>
      <th>5637133</th>
      <td>160543</td>
      <td>1976</td>
      <td>155673</td>
      <td>1237</td>
    </tr>
  </tbody>
</table>
<p>5637134 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b03795b1-77d0-4028-b636-1b465ce4cf6d')"
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
          document.querySelector('#df-b03795b1-77d0-4028-b636-1b465ce4cf6d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b03795b1-77d0-4028-b636-1b465ce4cf6d');
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
# generate JAX arrays (this will take several minutes)
start = time.time()
all_baskets = list(basket2idx.values())
items_per_basket, all_baskets, all_items, all_nonzero = dg.gen_data(df_train, 
                                                                    basket_idxs=None, 
                                                                    max_items=100)
duration = np.round((time.time() - start)/60, 2)
print(f"Duration of data generation: {duration} minutes")
print(all_baskets.shape, all_items.shape, all_nonzero.shape)

```

    Duration of data generation: 8.79 minutes
    (157604,) (157604, 100) (157604, 100, 1)



```python
# explore the number of baskets and unique items (movies)
print(f"Number of baskets: {len(basket2idx)}")
print(f"Number of movies: {len(item2idx)}")
```

    Number of baskets: 157604
    Number of movies: 6082


### Estimating the embeddings

Now we can simply define the parameters of the model and estimate the embeddings use our custom class.


```python
# create a dictionary with the model parameters and a random number generator
model_args, generator = be.gen_model_args(seed=92, 
                                          num_items=len(item2idx), 
                                          num_baskets=len(basket2idx),
                                          embedded_dim=50, 
                                          init_mean=0.0, 
                                          init_std=0.1/np.sqrt(50),
                                          rho_var=1.0,
                                          alpha_var=1.0,
                                          zero_factor=0.1,
                                          num_epochs=8000,
                                          batch_size=int(0.05*len(basket2idx)),
                                          ns_multiplier=50,
                                          num_ns=10,
                                          print_loss_freq=500,
                                          save_params_freq=500
                                          )

```


```python
# initialize parameters
params = be.init_params(model_args, generator)
print(params.rho.shape, params.alpha.shape)
```

    (6083, 50) (6083, 50)



```python
# create optimizer from OPTAX library
optimizer = optax.adam(learning_rate=0.01, 
                       b1=0.9, b2=0.999)
optimizer
```




    GradientTransformation(init=<function chain.<locals>.init_fn at 0x7f919a5cd560>, update=<function chain.<locals>.update_fn at 0x7f919a5cd5f0>)




```python
# initialize current optimal state with initial parameters
opt_state = optimizer.init(params)
```


```python
# train the model!
final_params = be.train(params,
                        optimizer,
                        opt_state,
                        items_per_basket, 
                        all_baskets, 
                        all_items,
                        all_nonzero,
                        model_args,
                        generator,
                        output_path="./")
```

![imagen.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjkAAAFrCAYAAAAzRZgMAAAgAElEQVR4XuydB3wVVdqHXyANQgu9hd6k9yIKKFJEFAWRFcu6KquydvFbseCirm1RsIAiithRFAuoCIqAIiAgvffeOyQkJPC9/5NMuLm5yZ2b3DI3+Z/98duYzJ0585xzZ555z3vOFDqvRVhIwJ3AlNul9tDZWbh0HblFJvQjLr8TMLw3y+Bpc2RYI7/vPYA7XCMvdOkj43e6HSJ+sEybM0zC6lQCSIm7JgESCA2BQpSc0IDnUUmABEiABEiABAJLgJITWL7cOwmQAAmQAAmQQIgIUHJCBJ6HJQESIAESIAESCCwBSk5g+XLvJEACJEACJEACISJAyQkReB6WBEiABEiABEggsAQoOYHly72TAAmQAAmQAAmEiECBk5wpt9eWoZs5vRX9bc0LXaQP5v52HSlbOC88T19BsswTPn6YBEiABAJCILCSk81aKxLCNTT8JznZrA8iXWXklgmSaSkZNw7xg6fJHPfFUNa8IF36jJeM5UY8iof7MT0cy4duEtwb8xS5vfZQmS3xeVoLJqPO6eeZ7bo9tnj6AMvLpsFk6b8+7L/z555IgARIwIkEgiA5zlrgzH83iDThmN7Lg7C4trT7Im/pN1+9018QnfTf1clYaC9dCDKJTrrg1LkQdTHnMjtvohOMTmnVc/DgzTJ+vORacrK0Xbo8uotOmnDk/jjBYJKXY/ivD+elFvwsCZAACTifACUn121kR3LSZGWzW+Qm7SZcJz3ik1VeTJXc5CjzZ6xKe95/rk8pAB809d74gBkOy5N8GBGcLr3cVgT2LD7OEmt/Y6Xk+Jso90cCJJBfCYRecszNXPSG30emmeGMtJJ1SCfr8JDnoYqch3TMDUI0GtJnmstrC3ITDbEhOR5vzBfql1b/NFFRAJlel5AW/bjAIaPerrkz1jCYj8N/7kM+nobPPHPyx1BTLiMs2bz2wF3+PHLy+dtrte1I6TV9qHllARi9JfdkyWHyG0tP55cpwmcN93k+mSxt6D5U7Gn408NwMl/b4XNn4QdIgAQcTMAhkpOmNhkX2CzDEB6Gb9K3yXxxT98u001ff3e7yIR0ObDk4UJeULp0iK/JyJ5yctxkKUPgrBwdS2hGigxNj/D0/dYtQuF2gzXDUw2zDI2Zm+v0XjKy13QZmhEV8rWnZR8J8h+nC3XKUyQnu3c7ZWKcVU7Sju6rxGYW0aHrwVpFp84DMsfIsadIUR5ZepWcCxy9RXKyRv08RAuzDJH62ne4PQmQAAk4n0AQJMeKzbjAcH2q9JhXkfmi7HmoBiM6mWdK2clR8XSDyG7/vjZf2vEvRDsy7TdTVMflhthgpMtN0zWq48pAMg17uUYr8lZ3Lzdmt1loeTuWNZsrl5Ec8ZSnZEU3LInxnNxsp19kbuvM4quGeSHHZ112L9LMI0u/SY7nyGDa8CcChunCHbYvBPX1W8ntSYAECjKBIEiOl/yIbC62rjKyzhpicp/m7OEpfrxLYq6nhg2k5Kh2Zc7BseqXFrhxmXXlciNqmJ5rMrKXTB/qmnPiOhwm6ZGctOET12RnK6KTuzc+5/HG7OM3J0+RHHMs9yEblZtMbLM5H5+jFjlIdrZykEeW/pIc91llmdrIdbgxcySSw1Q+dmZuTgIkEBYEnC05yJ1Rsck2z8KhkjPbilRZNxz3nBmPuRbuQyqZb5rW8JH7zcjb0EXOvTCPN2Yfu3jeJSfrATNLXnbn42uCdvhLzoWZet4bKWNoUjf1uLyB911wCxIgARJwJAGHSk7mpF6fhqu8LPQX0EhONlPBnTu7KtwlJ+vQjEchLkiRHPdoog+XnbwJsw8H4qYkQAIkECQCjpScLDkUVkTEWy6Pp+1w0XdPPPZzrklaW3lKevaQh+JpnRw7idYeclJ8zzVx71VOk5wLw1Feh088tjWaAbknsy8ksUtuEstDEMlxFzGXYSdf1wGyZnx5ZZipO2SzlEGQLkQ8DAmQAAkEgkAQJGd21nq7Dt94WhXZ45Ro93yM7KYze8jbcFmB2H+RHPfjuMwOcztjW6v02lkV2T0nxcep41a1XIcnMlf1wpCZ3zjllCPiPq05hxu70cj0afVpdc5hOrud6dM5fpvsS44/WWbuJ2iLtGUV3JcXyMoi6zCTe58zp+vC29vfA3Gx4T5JgARIINgEAis5ds6GszzsUOI2JEACJEACJEACPhKg5PgIjJuTAAmQAAmQAAmEBwFKTni0E2tJAiRAAiRAAiTgI4HQS46PFebmJEACJEACJEACJGCHACXHDiVuQwIkQAIkQAIkEHYEKDlh12SsMAmQAAmQAAmQgB0ClBw7lLgNCZAACZAACZBA2BGg5IRdk7HCJEACJEACJEACdghQcuxQ4jYkQAIkQAIkQAJhR4CSE+Amc19Zli9ADDBw7p4ESIAESIAE0gkEWHKyvkCx4JL39U3YDiFlvSIhp1dI2NnG5unk9OoGj68i0P1mekeTp9eE6Da5ksssr6Rwf1P8hZPKqJv7qyqwiYc6+fZeqczwbL/ewiZzbkYCJEAC+ZUAJSdoLRtukpP+bq6ug2Xw5vEyXgbLtDnDpFEmXna2sQvY2tdI2TKhn8cPGZGY3stDPVw2N0KB1z1NEM978a0+F94blc2LPjNeuDpY6owfL7M9SY77IbO8RNRundLf3+Xygtk0uRIZPG2ODMvcOPZ3yi1JgARIIJ8SoOQErWHDSXLSbugbH9giE/pl9xZvO9vYhWvvDdjBlRwPdc/ynjW06WtSzwiGd0lzifsYvuPrZC90Hsl5fM+bPXZ2W4LbkQAJkEB+IuAQybHzhvH0i/lOF/wehlCyDmnk8LZqz/GCbG9AacMErkMWWeuU/bBI9pJj/43f7sfL7tys7Xw9d09AspMc123tbJPD18ZEQ6ZLLy/RCGdIzuzMw2MZp+W75EzvNU3m+BB+8Xj+GUNq2Q+l5acLFs+FBEiABHwhEHrJyQj3X7jgW6LimrfgSQSynKif3mjueQggq6RMub2LrB/qMkyQ4zBEXiXHw+ezPd8wkxxriGmkyNChszOa1V0YPeXkZMlt8ZD/kqt8HA/fopyHhuxLTtp+6vg8pGa+A+IS/cnEbTOHrHy58nFbEiCBAkEg5JLjWV7cQ/D2QvK5vXlkbWlfhML10zkNSeVNcjxHMdK4+BoR8K1n24nS2Nkm+6N6TNr1IL+epXZ2zknFVqTDTq5MTmC87seL5LjJl++Jx5m/A5n6g5/k3rd+wa1JgARIwPkEQiw52d+ks9zUXW8S2c70cR32yttQjbt8ZXmKRttmmX2T1uCeIwd5k5zMM2rcOlZeb+A59lM7AmNnG2+SkzWyYSd6Z2ebvMuv1a9yGhKyH8nR9GHN5RlqL0nZBZvVB0dqPMdjRCfPydbOv2CxhiRAAiTgC4HwkRzrrDKJRXY3ncy5K74/NVsCY+WJeJgK7/HJPnCRHDs3c18a3v62dgTGzja+S44dOcmaJ5X1OHmbgWQJjjdp9lFcchF9sSJenofxfB/+st8HuCUJkAAJhCeBEEtO1imxaRhtDE+lS0adkZgBlB18G/vJtt1cokwNRkrt1+plnrrs8SYVOMmxc8MPTBe0IzB2tsmhdtkkHnsXO3vt630/XvrPTm+Cg8/nRnJ8nOrO2VWB6eLcKwmQQL4lEHLJsRZKc306tfXkbetJOG/Ttq0hs8F1xsv0em4zYbJI1oWhMl+Hq9wF5kKCrWukKn3/OS3Kl9FNHZh4nOMsIA+SZGMtmbQoTs4C4imJ/QImzOoaLzvFc0TQzv4vXBl8kJyc8o0yOHk6r6ycbH1X8u3liydGAiRAAjkTCIrkzM5SB7cLuNeVZd2nmGOHWW8CnvJW8jazJudcDPfZPl1HTpN6r2VOBM4+lyaHqejIsekzzcOidvY4ZETCbEUgPHeQ7FYXxtbW8J+dbbJIAFpusKep096nx2dh6SEXKUudcpTCHMQ0m5WTzfm47DP79r3QP7Nuk4OYZRzX2/IAFllOHedFngRIgASyIxBgySF4ErhAIMeoSohAObFOIULBw5IACZBAviNAycl3TerUE/JluC1Y5+DEOgXr3HkcEiABEsj/BCg5+b+NQ36GGcM1AZ3q7ttpOrFOvp0BtyYBEiABEvBGgJLjjRD/TgIkQAIkQAIkEJYEKDlh2WysNAmQAAmQAAmQgDcClBxvhPh3EiABEiABEiCBsCRAyQnLZmOlSYAESIAESIAEvBGg5HgjxL+TAAmQAAmQAAmEJQFKTlg2GytNAiRAAiRAAiTgjQAlxxsh/p0ESIAESIAESCAsCVBywrLZWGkSIAESIAESIAFvBCg53gjx7yRAAiRAAiRAAmFJgJITls3GSpMACZAACZAACXgjQMlxIbRp0yaZOXOmbN68WWJjY+WWW26RunXr5sjwyJEjMmPGDNm2bZucPXtWUlNT5cyZM1K+fHl55JFHvPHn30mABEiABEiABAJEgJLjAnbVqlXyzTffGGFZunSpvPbaa3LJJZfkiP7o0aPy66+/yo4dO4zk7Ny5UzZu3CjlypWTjz76KEDNxt2SAAmQAAmQAAl4I0DJcSEEYdm1a5eJxDz66KPy3HPPGclJSUmREydOyNatW83PhQoVkkqVKknFihUlOjo6E+Pp06fLb7/9JlWrVpUhQ4Z448+/kwAJkAAJkAAJBIgAJccNbFJSkpGZe+65R5599lkjOYcOHZIFCxbIxx9/LCdPnjSS061bN7nmmmukTp06GXs4d+6cvPnmm7Ju3ToZPHiwtGzZMkDNxt2SAAmQAAmQAAl4I0DJsSE5P//8s3z55Zdy7733Snx8vJGgb7/9VkqUKCEPP/xwxh4OHjwoI0eOlISEBHn11VclMjLSG3/+nQRIgARIgARIIEAEKDleJKd9+/by2WefyYgRI6RNmzYSFRUliPbs379fOnfubKI9VkECMvJzatSoIXfddZeJ+LCQAAmQAAmQAAmEhgAlx4vktG3bVj755BMZO3asPPnkkxlbY2iqevXqRnys8vzzzxsB6tmzp1x88cWhaVEelQRIgARIgARIwBCg5HiRnA4dOshXX31l8nEwBFWzZk0zDHXq1Ck5f/68GbLC/x8/flyGDh0qDRs2lNtuu83MrmIhARIgARIgARIIHQFKjgt7rHmzYcMGWb9+vYwePVr+8Y9/yGWXXWamhyMvB4LTtGlTIzmYLo7ZVY0bNzZr4/z555/yzjvvmITkm2++OXQtyiOTAAmQAAmQAAkwkuPeBzCDaty4cbJy5UoTmUGU5oorrpArr7xSYmJizJAVRCg5OVkuuugiufrqq6V79+5mWvmHH34ohw8fNnk6yONhIQESIAESIAESCC0BRnJc+CMigwgN8m0wBIXE4SJFikhERITZCn/D71EKFy5sfo+/u/4N/239LrRNy6OTAAmQAAmQQMEmQMkp2O3PsycBEiABEiCBfEuAkpPetFOmTJF58+aZISoWEiABEiABEvCFAFbKr1Klitx///2+fIzbBpgAJScdMKZ/z58/3+TYsJAACZAACZCALwSWL19uXgmEJUdYnEOAkpPeFphNhfdTDR8+3Dmtw5qQAAmQAAmEBYEvvvhCZs2aJW+//XZY1LegVJKSQ8kpKH2d50kCJEACASNAyQkY2jztmJJDyclTB+KHSYAESIAERCg5zuwFlBxKjjN7JmtFAiRAAmFEgJLjzMai5FBynNkzWSsSIAESCCMClBxnNhYlh5LjzJ7JWpEACZBAGBGg5DizsSg5lBxn9kzWigRIgATCiAAlx5mNRcmh5DizZ7JWJEACJBBGBCg5zmwsSg4lx5k9k7UiARIggTAiQMlxZmNRcvwkOXhx55wNB+Vs6nmpV6G41CwX68wWZ61IgARIgAT8ToCS43ekftkhJcdPkoPdPDdtjZw4kyK9mlSSyxtW8EsDcSckQAIkQALOJ0DJcWYbUXL8KDl3f7RE9p5IlIFt4mVQ+xrObHHWigRIgARIwO8EKDl+R+qXHVJy/Cg5d36wWFbvOS4DVHIe7l7fLw3EnZAACZAACTifACXHmW1EyfGj5Nw+cZHMWndA+reqJq/c0NyZLc5akQAJkAAJ+J0AJcfvSP2yQ0oOJccvHYk7IQESIIGCTICS48zWp+QEQHL6taoqIwc0l8KFCjmz1VkrEiABEiABvxKg5PgVp992RskJgOS0qh4nd3WpLT0bV/JbQ3FHJEACJEACziVAyXFm24Rccg4fPixz5syR7du3S0REhHTt2lWaNm2aLa3k5GTZvXu3fPXVV9K6dWtp3ry5lClTRk6fPi3r1q2TxYsXC7apXbu2NGvWTOLj422RHz16tJw4cUKGDx9ua3tPG1k5OeWKR0v3RhXlhX7Zn0euD8IPkgAJkAAJOI4AJcdxTWIqFHLJ2bFjh7z33nuyfv162bp1qzzwwAMyaNCgbGnt27dPvvzySxkzZozcfPPNcsMNN0idOnVky5Yt8s4778ipU6cEC/NFR0fLxRdfLP369TPy5K34U3KiIgpLm5px8umdHbwdln8nARIgARLIBwQoOc5sxJBLDqQEggMRef7556Vv377ZSk5iYqIsXbpUXn75ZTl37px0795devXqJZUrV5bff/9dnnnmGfnoo4+kfPny8vrrr8vx48dl6NChUrFiRa/0/Sk5kUUKSZsacfLZPzt6PS43IAESIAESCH8ClBxntmHIJQdYICwYgnrsscfkqquuylZyVq5cKbNmzZJCmtCLnzFc1a1bNylcuLDMnDnTyNLTTz8tpUuXlsmTJ8vq1aulU6dORoY8FUR88A/ltddek5MnT/pluAr7a1qtlEy99xJntjprRQIkQAIk4FcClBy/4vTbzsJGciAgP/74oyxYsEAefvhhE81p1KiRkZykpCSZPn26pKSkyJAhQ6RkyZLmv//66y+pW7euGdJyL8gFwjY//fSTJCQkGMnq3LmzvPTSS7mGa+XkUHJyjZAfJAESIIGwJEDJcWazhY3kQEZWrVol9evXl8svv1yeeOIJadiwoZEcDGNBWBDRueuuu6REiRIyY8YMWbJkidSoUcNjZAhiBLHZtWuXpKamyjfffCOlSpUyQ165LZSc3JLj50iABEggvAlQcpzZfmEjOciZmT17thEbyAiiOsi9QQ4PZlCtWLFCEJ155JFHzN+/++47Wb58uZlhhW28FX/m5DCS4402/04CJEAC+YsAJceZ7Rk2kjNlyhQz1Rzl7NmzMn/+fImLizOzp5Cbg5lZU6dOlVGjRpkp5RMnTjTT0vv37y+tWrXySp+S4xURNyABEiABEsiGACXHmV0j5JKDYaM9e/bItm3b5JVXXpEuXbrIgAEDTF6NlRSMnyMjIzMIYk0c1+EqDElhaOrxxx+XYcOGmcRjSFHRokXN7KrY2Fiv9Ck5XhFxAxIgARIgAUpOWPWBkEsOFvD7z3/+I4sWLZJDhw5JTEyMXHrppdK7d2+TSIyCGVdVq1bNAItE4aeeekoaNGhg8nOQXHzgwAEjNlhzB0nKV1xxhcnFwVo5dgolxw4lbkMCJEACJOCJACM5zuwXIZccrE6MXBpEdCA1SB5GBAbRFyuSg59dF/TDlHN8JioqSooVK2aiPEgexpo7WBsHf8fvkYCMfdkplBw7lLgNCZAACZAAJSd8+kDIJccpqCg5TmkJ1oMESIAEwo8AIznObDNKTnq7UHKc2UFZKxIgARIIBwKUHGe2EiWHkuPMnslakQAJkEAYEaDkOLOxKDmUHGf2TNaKBEiABMKIACXHmY1FyaHkOLNnslYkQAIkEEYEKDnObCxKDiXHmT2TtSIBEiCBMCJAyXFmY1Fy/Cg5gz9cLL+sPSDn9M3mfAu5Mzs8a0UCJEACgSBAyQkE1bzvk5LjR8m5/7OlMmPNPjlz9hwlJ+99k3sgARIggbAhQMlxZlNRcvwoORPnbZNJi3bIun0nKTnO7O+sFQmQAAkEhAAlJyBY87xTSo4fJefw6WR5dcZ6+WThDkpOnrsmd0ACJEAC4UOAkuPMtqLk+FFy8BqKF39cJ+PmbqHkOLO/s1YkQAIkEBAClJyAYM3zTik5lJw8dyLugARIgAQKOgFKjjN7ACXHj5KDXb3w41oZN4eRHGd2d9aKBEiABAJDgJITGK553Sslx++So8NVczbLRZVLysd3tJPSxaKkSOFCeW0nfp4ESIAESMDBBCg5zmwcSk6AJKd2uVh5vl9TaRFfWmIiiziz9VkrEiABEiABvxCg5PgFo993QskJkOSULxEtA9pUk1s61JAKJWIYzfF71+UOSYAESMA5BCg5zmkL15pQcgIkOdhtVERhGXVDc+lUt7wOW0U6swewViRAAiRAAnkmQMnJM8KA7ICSE0DJQS5O3+ZV5K4udaRBpRIBaUDulARIgARIIPQEKDmhbwNPNaDkBFhyul9UUS6pV07a1SojNTVPJ6pIYWf2BNaKBEiABEgg1wQoOblGF9APUnICKDmYVFW/YgkpHhMhLePjpGfjitKmZpmANih3TgIkQAIkEHwClJzgM7dzREpOACUHuy4aVUSSU85JQx2uuql9dRnUvoadduE2JEACJEACYUSAkuPMxqLkBFhyrGZHRGdQu+pyW6eazuwJrBUJkAAJkECuCVByco0uoB+k5ARJcupWKC79W1WTm3VKeQkdvmIhARIgARLIPwQoOc5sS0pOkCQnQhN0YqMjNCcnTt69tY0UKsRVkJ35lWCtSIAESMB3ApQc35kF4xOUnCBJDg4DrWlYuYSRnIqligrEh4UESIAESCD8CVBynNmGlJwgSg4OVbpopHTTaeXDejeUcsWjndkrWCsSIAESIAGfCFByfMIVtI0pOUGWHBwuWldCnvFQF6lRtljQGpoHIgESIAESCBwBSk7g2OZlz5QcSk5e+g8/SwIkQAIkoAQoOc7sBiGVnOTkZElMTJSzZ89K4cKFpUSJEhIZmfUdT/j7mTNnzHYoSNqNjY012+Jn7OPUqVNSpEja276xr6JFi0p0tP3hoNGjR8uJEydk+PDheWqpF35cJ+PmbM5xH4zk5AkxP0wCJEACjiNAyXFck6T5wnktoara77//LuPGjZMlS5ZIXFycvP7669K6dess1Vm0aJFMmjRJFi9ebASmePHicv/990u7du2kVKlS8vHHH8uTTz4pNWvWNJ8tV66c3HnnndKrVy/bp0bJsY2KG5IACZAACbgRoOQ4s0uEVHI2b95sxOXYsWMyYcIEIznt27fPQurgwYNy6NAhSUhIkKSkJNm2bZt8++23MmzYMGnRooW8//778vbbb8tbb71lPosIT5UqVaRs2bK2qVNybKPihiRAAiRAApScsOgDIZUcSMvhw4dl//798uCDD8orr7ziUXJSUlJMBAf/MCyFyM9jjz1mtr/44otl4sSJJprz888/5xo6JSfX6PhBEiABEijwBBjJcWYXCKnkAAlybTZu3ChDhgyRkSNHepQcbIdIzoYNG2TVqlWydetWOX36tPlMw4YNTRQIkvLEE08YEapXr57UqlXLDGVlV5DHg+Nu2rTJ5PrMnDlTKleuLM8++2yeWoo5OXnCxw+TAAmQQFgSoOQ4s9nCRnK2bNkis2bNkl9++UWOHDki119/vfTp08eIybRp02TMmDEmJwfCUq1aNenZs6fJ74mKivJIHkNkc+bMMf8gPBCotm3byosvvpinlrIvOZ11Cnlsno7FD5MACZAACTiDACXHGe3gXouwkRyr4idPnjR5PKNGjZKnnnpK2rRpYyQF/5C8vHfvXhk6dKi0bNlSbr31VqlUqZIt8sEervrpwc5Ssxwlx1bjcCMSIAEScDgBSo4zGyjsJAeTwY4ePWqiOJCcyy67LGOqOKaTnzt3Tl5++WUzbNW9e3cjO3ZKKCQHkRy+wspO63AbEiABEnA2AUqOM9snLCQH+TgxMTFm6jgiNmvWrJE77rjDRHOs2VgYlsI6OYj0PProoyYn58Ybb5QaNWrYIu8vyVmz94R8unCHfLxge7bHjdIVjz+8vZ00r1Zaikalre3DQgIkQAIkEL4EKDnObLuQSs7atWvlxx9/lJUrV5rEX0RlevToYXJrEJGJiIgweTLz58+XZcuWmQgOIjmYlYXFADEcVaxYMcE6OtYaOkhkxj9Ecbp27Wr+bqf4S3KSUs7Jl4t3yhPfrMr2sJFFCsurNzSXS+uVk9LFPOcM2akztyEBEiABEnAGAUqOM9rBvRYhlRwk+86ePVt27dplIjSI1jRt2lSqV6+eITlYB2f16tWyfPlyOXDggInWYDVjCBFkCJ+DAEFyMFwFCWrWrJlZP8duPg6g+EtysK+vluySRyYvz1Fy/ntdE7m8YQW+pNOZ3wvWigRIgAR8IkDJ8QlX0DYOqeQE7SxtHMifkvP9yr3ylEZyjpxO9njkiCKF5MFu9eS6ltWkalxRG7XjJiRAAiRAAk4mQMlxZutQctLbxZ+Ss2jbERn76yb5df1Bj61eWCNOLauXlid6XyStasQ5s2ewViRAAiRAArYJUHJsowrqhpScEEgODlmkcCF555bW0u2iikFtcB6MBEiABEjA/wQoOf5n6o89UnJCJDk47MPd68t1rapKfJy95Gh/NDj3QQIkQAIk4H8ClBz/M/XHHik5IZSca1tUlVs61pDWHLLyR1/mPkiABEggZAQoOSFDn+OBKTkhlJzLGlSQf3SqKZ3rl3dm72CtSIAESIAEbBGg5NjCFPSNKDmBkpzZmni8znPisdXKXVRubr+kluD/WUiABEiABMKXACXHmW1HyQmA5CzG7KrZm2XWugM5tjolx5lfCtaKBEiABHwlQMnxlVhwtqfkBEBy/tpxVMbN3SI/rdpHyQlOP+ZRSIAESCCkBCg5IcWf7cEpOQGQnBOJZ2XlnuMyddkeuaJRRV0zZ7NAfNxLu1plZPCltaW7bsNCAiRAAiQQvgQoOc5sO0pOACTnnL5a4nRSiuw8kig1yhWT+z5d6qhIFm8AACAASURBVHHoqmxslDxwRT25pUNNvo3cmd8P1ooESIAEbBGg5NjCFPSNKDkBkBz3Vrzjg0Xyy1rP+TlDezSQe7rWMYsDspAACZAACYQnAUqOM9uNkhNiyXmkR30Z0rUuJceZ3w/WigRIgARsEaDk2MIU9I0oOZScoHc6HpAESIAE8hsBSo4zW5SS4wDJuadLXcGbyVlIgARIgATCkwAlx5ntRskJseTg/VXIyYksUtiZPYS1IgESIAES8EqAkuMVUUg2oOQEQXLu1MTjn7NJPKbkhKTf86AkQAIk4FcClBy/4vTbzig5QZCcwR8uVsnZLzqzPEsxktNFIzkRjOT4rVdzRyRAAiQQZAKUnCADt3k4Sk4QJOef6ZJTPCZS4opFyvbDCRnN81C65ERRcmx2WW5GAiRAAs4jQMlxXpugRpScIEjOI18sk+n6iocGlUpIp7rl5I1Zmy5Iji4GeLfm5ERHFHFmD2GtSIAESIAEvBKg5HhFFJINKDlBkJwZq/fJxD+2SRld4fjGdtXlpncXZjT2gyo5SDym5ISk//OgJEACJOAXApQcv2D0+04oOUGQnFNnUuRIQrLJydl7PFH+9s6CjIbEax2QkxMTyUiO33s3d0gCJEACQSJAyQkSaB8PQ8kJguRYbXI29Zy+qPOYDBw3/4LkdEsbripKyfGx63JzEiABEnAOAUqOc9rCtSaUnCBKTkrqefM28htcJOd+lRwMV1FynPkFYa1IgARIwA4BSo4dSsHfhpLjAMm5W4erikVxuCr43Z9HJAESIAH/EKDk+Iejv/dCyQmx5ODlnHd3qS0li0b6u225PxIgARIggSARoOQECbSPh6HkhFhyBraNlzsuqSX1K5bwsem4OQmQAAmQgFMIUHKc0hKZ60HJCaLknNPpVct3HpPrxv6R0QqlNILTv3U1Gd6nkTN7CGtFAiRAAiTglQAlxyuikGxAyQmi5OBQa/eekNsnLpIDJ5Mk9Vzaex6ubVlVRg9sEZIOwIOSAAmQAAnknQAlJ+8MA7GHkEvO4cOHZc6cObJt2zaJiIiQyy67TJo2bZrlXPfv3y+rV6+WdevWSaFChaRw4cLStm1bqVevnpQoUUJOnz4ta9eulcWLF0tycrLUqVNHmjVrJvHx8ba4jR49Wk6cOCHDhw+3tX1uN9pzLFHGzt4s3y7bLSd1/RyUPs0qy/8GNOcMq9xC5edIgARIIMQEKDkhboBsDh9yydmxY4e89957smHDBtmyZYs88MADMmjQoCzVhQTNnz9f/vrrLyM5kJoqVapI7969pXnz5rJ582Z55513JCEhQRfdOy/R0dHSsWNH6devn5EnbyVYkpOQnCqr9xyX+z5bKvuOnzHV6tWkkvz3uiZSNjbaWzX5dxIgARIgAQcSoOQ4sFG0SiGXnFOnTsn69euNiDz//PPSt29fj5KD6AxKZGSknDt3zkR0RowYIX369DEi8/vvv8szzzwjH330kZQvX17eeOMNOXbsmAwdOlQqVqzolX6wJAcVSUo5Jz1Gzcl4UWfPxpXk2WubSIUSlByvDcUNSIAESMCBBCg5DmwUJ0gOsEBadu/eLY899phcddVVHiXHwocozdmzZ2X58uXy0ksvGSm6+OKLZebMmUaWnn76aSldurRMnjzZDG916tRJunfv7pV+KCWnR+OK8mzfJlKxZIzXenIDEiABEiAB5xGg5DivTVCjkEdyfJGcXbt2yS+//CKffvqpIEfn5ptvluuuu04SExNl+vTpkpKSIkOGDJGSJUua/8bQFnJzBg4c6JV+KCWne6OK8kzfxlK5VFGv9eQGJEACJEACziNAyXFem4Sd5CDfZs+ePbJp0yaTZIyfEfkpU6aMkRokI991110mEXnGjBmyZMkSqVGjhsfI0PHjx2XevHkmz+fMmTOyatUqadGihbzwwgsBbyn34arm8aXlrs61pXfTygE/Ng9AAiRAAiTgfwKUHP8z9ccewyqSY50whrf27t0rTz31lHTu3Flatmwps2fPFszUeuSRR6RUqVLy3XffmSEtzLDCkJZ7sWZjIbcH+T74fPXq1eW5557zB9cc9+EuObXLxcqANtX0HVZ1A35sHoAESIAESMD/BCg5/mfqjz2GjeQkJSWZWVMxMTHm/xGJQdSmS5cucuWVV5qIzNSpU2XUqFEmsjNx4kTZvn279O/fX1q1auWVVSiHq6qXKSbX6Vo5D3Wv77We3IAESIAESMB5BCg5zmsT1CjkkoOhIgw7YYr4q6++aiIzAwYMMHk1VsHP+/btkyNHjpip4YjkYPtPPvlErrnmGpOXg6Gpxx9/XIYNG2YiOV9//bUULVrUzK6KjY31Sj+UklMtrqj0bVFFHu3Z0Gs9uQEJkAAJkIDzCFBynNcmjpAca0YUFvE7dOiQidRceumlJjqDRGIU5N1gu88//1z+/PNPM40csnPnnXeamVNYL+fAgQNGbLDmzsmTJ6Vbt24mFwczr+yUUEoOpo5b08jt1JXbkAAJkAAJOIsAJcdZ7WHVJuSRHOTDIEKDiE5qaqpJHoboIPqCYSkU/Ixp45AXbIfFAPEPEZ5ixYqZNXbwWay5g1WLEenB75GAjH3ZKaGUnCKFC0kLTT7+6h57QmbnfLgNCZAACZBA8AhQcoLH2pcjhVxyfKlsILcNpeTgvBpUKiEf3N5OyhePFkgPCwmQAAmQQPgQoOQ4s60oOentEkzJSdYVj297/09Zpm8kx2seUGrpDKuX+jeTZtVKSUxkEWf2FtaKBEiABEjAIwFKjjM7BiUnBJJzNvWcDP92tfy8Zr8cPJVkahCvM6ye6H2RdK5fTopFeX/XljO7E2tFAiRAAgWTACXHme1OyQmB5KSo5Lw4fb18v2KP7E1/SWd8XDF5HJLToJzEUnKc+W1hrUiABEggGwKUHGd2DUoOJceZPZO1IgESIIEwIkDJcWZjUXJCJDkvaSRnmlskZ1jvhtKlQXlGcpz5XWGtSIAESCBbApQcZ3YOSo6DJOcxlZyu9VVyopmT48yvC2tFAiRAAp4JUHKc2TMoOaGSnJ80krP8Qk5OueJRcn3ravqizjoSFxvlzN7CWpEACZAACXgkQMlxZseg5DhEcqIjCkvdCsVl/K1tpErpos7sLawVCZAACZAAJSeM+gAlJySSc15e+mldpkgOqhGlojPjoc5Ss6z3d22FUR9jVUmABEgg3xNgJMeZTUzJCZHkvKySM3X5Xp1CnpjRMyg5zvySsFYkQAIk4I0AJccbodD8nZITIsn5X7rk7KHkhKbn86gkQAIk4EcClBw/wvTjrig5oZKcGRrJWbZX3CVn+gOX6iseiusLSP3YytwVCZAACZBAQAlQcgKKN9c7p+Q4SHLwYs67OteWgW3jpQbzcnLdqflBEiABEgg2AUpOsInbOx4lx0GSg+BNxZIx8r8BzeXSeuXstSC3IgESIAESCDkBSk7Im8BjBSg5IZKckTpc9Z3bcJXVQphG3r1RRWf2GNaKBEiABEggCwFKjjM7BSUnRJLzyoz18q0uBrjn2IXZVVYXeeeW1io5lZiX48zvDGtFAiRAApScMOkDlJxQSM658zJuzmb5YtFO2X4kIUtXebFfM7mqWWUpEcPXO4TJ94jVJAESKOAEGMlxZgeg5IRAcs6fF9l2+LRMWrRDflq1T3/OLDp1yxeXIZfVkX6tqjmz17BWJEACJEACmQhQcpzZISg5IZAcHDJVozlr9p4wEZ1pK/Zm6R2P9mwg/7qsrjN7DWtFAiRAAiRAyQmDPkDJCZHk4LBnzqbKKzM2yPjftlBywuDLwiqSAAmQQHYEGMlxZt+g5IRQcpJTzslITUB+Zy4lx5lfD9aKBEiABOwRoOTY4xTsrSg5oZScVJWcnyg5we70PB4JkAAJ+JsAJcffRP2zP0oOJcc/PYl7IQESIIECTICS48zGp+SEUHLO6TSrl6evl7c1+di9MPHYmV8Y1ooESIAEPBGg5DizX1ByQig5OPTL09fJ2NmUHGd+PVgrEiABErBHgJJjj1Owt6LkhFhyxs7epNPIt8jxxLOZ2p6RnGB/FXg8EiABEsg9AUpO7tkF8pOUnBBLzi9r98sH87fL3A0HKTmB7OncNwmQAAkEkAAlJ4Bw87DrkEpOcnKyJCQkyNmzZ6Vw4cJSsmRJiYyMzHI62O7MmTNmOxRsGxsba7YtVKiQ+dupU6fMzyj4/2LFiklMTIxtNKNHj5YTJ07I8OHDbX/GHxti9eOvl+6Sh79Ynml3Q7EYYNe6fH+VPyBzHyRAAiQQYAKUnAADzuXuQyo5v//+u4wbN04WL14scXFx8sYbb0jr1q2znMrcuXNl8uTJsmzZMilSpIiUKVNGhg4dKs2aNZPixYvLlClT5L///a8RGwgQtvnXv/4l/fv3t40lVJKDCn69dLc89PmyTHW9v1s9uadrHSkaWcT2OXBDEiABEiCB0BCg5ISGu7ejhlRyNm/ebATn2LFjMmHCBHn99delffv2WeoMuTl69KhERUVJSkqKrFq1SpYsWSIPP/ywNGnSRCZNmiTvvvuuPPHEE1KiRAkjOtWqVZMKFSp4O/+MvztNcjrWLis3d6whVzWtbPscuCEJkAAJkEBoCFByQsPd21FDKjkYqjp8+LDs379fHnzwQXnllVc8Sg4EB9EZRGowZLV69Wq57777ZOTIkdKpUycjOZ999pkRJUSEIDm+FqdJTo2yxeSGNvF8f5WvDcntSYAESCAEBCg5IYBu45AhlRzUD/k0GzdulCFDhhhp8RTJcT0P5M0sXLjQCNGzzz4rbdu2NZIzZswYue2224wI1apVS+rVqydly5a1gSBtE6dJTsmikdJf30L+9NWNbJ8DNyQBEiABEggNAUpOaLh7O2pYSU5SUpKsW7dOPvzwQyMwgwYNkpo1a8ovv/wi7733nklcxnAW/tajRw/p2LGjkR5PBcnMiCAdOHBAUlNTBR0UycwjRozwxszvf/eUk4OD9G1RRUYPbMnkY78T5w5JgARIwL8EKDn+5emvvYWN5EBEtm3bJj/++KNMnTpVxo4dK9WrVzczrDDslZiYaBKSMbSF3JzKlSvLgAED5KKLLvLI6tChQ2Y/06ZNM5+H7HTr1k1efvllf7G1vZ/sJAf5OP8b0FyKRTH52DZMbkgCJEACISBAyQkBdBuHDBvJ2bdvn3z++eeCmVZPP/20NGzY0CQio5zHPGwtmDqOnydOnCh79+41Q1ndu3f3iAHbIZqDHB/8DGnC0Bn2HeySneR00OTjh7vXl3a1ygS7SjweCZAACZCADwQoOT7ACuKmYSE5iM588MEHsmXLFunZs6eJuERHRxupgaRgiAo/43cY0sKQU9GiRaVPnz7SqlUrWzidlpODSrepGSd3d6kjF1UuKRVLxkhE4bR1gFhIgARIgAScRYCS46z2sGoTUslZs2aNTJ8+XVauXCkzZ86Url27mlwa5NkguhIREWGiMdgGOTcHDx6Udu3aZcyguvnmm03+zaJFi2T27NlmBhakB8nJ2Nfll19utrVTnCg55YpHS8NKJSQuNkqeuuoiqaCiw0ICJEACJOA8ApQc57UJahRSydmwYYPMmTNHdu3aZXJqsEJx06ZNJT4+Xs6dO2ckp0WLFvLHH3/I0qVL5eTJkyY5GAXTxLHYHyRnxYoVMm/ePPN7fK5u3brSpk0bk7NjtzhRcqy6RxYpLD89eKnULl/c7ulwOxIgARIggSASoOQEEbYPhwqp5PhQz4BvGg6SU6tccc60CnhP4AFIgARIwHcClBzfmQXjE5ScdMqUnGB0Nx6DBEiABPInAUqOM9uVkhMmkjMdw1WM5DjzW8RakQAJFHgClBxndgFKDiXHmT2TtSIBEiCBMCJAyXFmY1FywkVyHkDicayZKs9CAiRAAiTgLAKUHGe1h1UbSk4YSE5EkULy/m1tpY7Oriql77SKjY5wZm9irUiABEiggBKg5Diz4Sk5DpCcacv3yKNfrpDEs6kee0kRXQTwpvbVpXKpotKhTllpGV/amb2JtSIBEiCBAkqAkuPMhqfkOEBy/th0SP43Y70s3XHMay8Z2rOB3HtZXa/bcQMSIAESIIHgEaDkBI+1L0ei5DhAcpJTzsmS7UfkxvELvbYdJccrIm5AAiRAAkEnQMkJOnJbB6TkOEBy8AqLlbuPyzVvpq3anFPxJDkHTyYJhrRKxkRIhK6OzEICJEACJBBcApSc4PK2ezRKjgMkB1VYpZLT543fvbYbJOdfXevK0YRkmbVuv7SsHiffLN0tCcmp0rl+eemi/1hIgARIgASCS4CSE1zedo9GyQlDyblZk5AXbTsiY3/dbBKRZ67ZL4kqOQPaxMuDV9Sz2/bcjgRIgARIwE8EKDl+Aunn3VBywkxy7upSWy6uU06+WLxTflixV2qWixUMV2H5nOtaVpVn+jbxcxfh7kiABEiABLwRoOR4IxSav1NywkxyMBxVWtfK+VannWNZQEtycBr9IDnXUnJC81XiUUmABAoyAUqOM1ufkhNmklM4fcXjc5qsDMmphUjOqSTR/6TkOPM7xlqRAAkUAAKUHGc2MiXHIZJzRhcCXL/vpNzz8V+y53iird7iKjmnk1KlQ+0yJienXa2ytj7PjUiABEiABPxDgJLjH47+3gslxyGSg2pgvZzuo+bK9sOnbbWzq+ScPJMicbFR0lWHs0YNbGHr89yIBEiABEjAPwQoOf7h6O+9UHLykeTgVBpWKiHTH+zs737C/ZEACZAACeRAgJLjzO5ByaHkOLNnslYkQAIkEEYEKDnObCxKTj6TnAYayZl23yUSUbiwmVbOQgIkQAIkEHgClJzAM87NESg5+Uxyautsq/G3tpbqZWMlkq94yM13gp8hARIgAZ8JUHJ8RhaUD1By8pnkREUUlvi4YjLxH20lvkyxoHQiHoQESIAECjoBSo4zewAlJ59JDk4HL+uc+VAXqV0+1pm9jrUiARIggXxGgJLjzAal5ISx5KDqccWi5HRyipl+bhUsGDjmppZSPDpCqmlUBwsGspAACZAACQSOACUncGzzsmdKTphLjqfGR8LxlU0qS1JKqv5/Jbm2ZTVNRGYWcl6+KPwsCZAACeREgJLjzP5BycmHkuPa1f7ZubYM6VpXSheLdGYPZK1IgARIIB8QoOQ4sxEpOflccga2jReITp3yxZ3ZA1krEiABEsgHBCg5zmxESo5DJadG2WISrTOlNuw/laeec1nD8vL3jjWla4MKedoPP0wCJEACJJA9AUqOM3sHJcdBkpN67rxMXrJTEpNTBT8v23lMpq3Ym6eeU6FEtPRvXU3+3athnvbDD5MACZAACVBywq0PUHIcJDnntS5HTycLUoRPJaXI2r0nZPqqfTJl6e5c9ytMJ7+meZWMl3aeO39ejugxonShwJJFmaeTa7D8IAmQAAm4EGAkx5ndIaSSc+DAAVm/fr0cPnxYoqKipH379lK2bNkspPbv3y/bt2+XI0eO6KsKCpltmzRpInFxcRIRESEJCQmyZ88esy+USpUqSa1ataRMmTK2qY8ePVpOnDghw4cPt/2ZQG+IaeF/7Tgqf3tnQa4P5S45mw6clIVbj0iV0kWlbc0yZpo5CwmQAAmQQN4IUHLyxi9Qnw6p5Pzxxx/y/vvvy/LlyyUxMVHeffddIzruZe7cufLDDz/Ihg0bzJ8iIyPl+uuvl8suu0zKlStn5Obbb7+VWbNmSWF9Z1Pt2rXN3zt16mS2tVOcKDmo95o9J6T367/ZOQWP22DmOCI5o//WUjSII+PmbpbPF+1UwYkzCcl1K5TI9b75QRIgARIggTQClBxn9oSQSs7evXtl69atkpKSIsOGDZNXX33Vo+SsWbNGYmJiTIQmKSlJZs+eLWPGjJHnn39e2rRpI59//rlMmjRJXnvtNalYsaI89NBDUrduXRk0aJBUqVLFFnnHSo4OWfV5/XfBMFNuy9UqOW/cmCY5//lulUxfvU/a1Sord1xSS1rEl87tbvk5EiABEiCBdAKUHGd2hZBKDuTm9OnTRnTuu+8+GTlypEfJSU7WPBUdpipSpIgRImwPgYGYYFhq6tSpsnr1ahk1apQZvkJ0CENhbdu2lW7dutki71TJQf7M15qTM+bXTSaXJjeFkpMbavwMCZAACdgnQMmxzyqYW4ZUcnCiZ86ckY0bN8qQIUOylRxXIIcOHZKffvpJvvvuOxP9QX7OzJkz5dixY/L000+bTTG0tXTpUqlfv74MGDDAI8/jx4/LvHnzZP78+aYOq1atkhYtWsgLL7wQTP5ej5WSel4OnDwjP2n05ZOFO2TTAd+nlF/VrLK89rcWUqRQYXk6PZLTnpEcr+y5AQmQAAnYJUDJsUsquNuFleQgMXjJkiUmUtO9e3fp0aOH7Nu3T3799VcjKo899pih9/PPP8uCBQtMbg4iPp4KIkhr166VdevWCSJFGAKrXr26PPfcc8FtARtHwzDTjiOn5fGvV8m8TYdsfCLzJnUrFJfujSoK3mk1d8NB2XLotFzesAKHq3wmyQ+QAAmQgGcClBxn9oywkRwkJiM6M2PGDEEU5tFHHzX5N0g6RiTn6NGj8p///MdQ/v7772XZsmXSoEEDk4Bspzh1uMqq+4nEs/LApKXy6/qDdk7H6zYYwrpTc3KaZ5OTA7HCO7BYckcAM+PAD7PbIJcsJEACgSeAIf3Dp5LM9652kFd5p+QEvn1zc4SwkBzk4SDnBrk3u3btkieeeMIkFCNHB1PHMXS1cuXKjJycCRMmCIa1MFMLM7DslPCQnGUqOQfsnI7XbTDj6o5LVXKqZU08Pq5Ctftooq6jEyEVS8ZIpK6pw+IbASzkeF5NEVP1wZCFBEgg8AR+Xrtfft94SMoVj5J7L68X+AO6HIGSE1Tctg8WUsk5d+6cSTxGTs79998vL774onTo0MEkGVsFIoNZWJh5hXVy7r33XmncuLH5M5KMUdC5MLsKicvly5eX//u//zP5OBiqqly5si0YTpecBF0F+d5P/5JZ6/wnOXeq5DTzIDnf6yrLw6as1CGuCvJoz4ZSqVSMWYEZM7wQlcBTEkvOBPq/9YfsP3FG/tGplhkWZCEBEgg8gX9/tcIskdG5fnn58PZ2gT8gJSeojHNzsJBKDhJ/sTYOhpY2b94sNWvWlKuvvlqaNWsmqampEh0dLb1795aPPvpIPv74Y8GigDVq1DC/h/yMGDFCmjdvbiTpm2++kWnTphlBatSokQwcONCsk2OJkDc4TpccDB/986PFMnPNfm+nYuvviOTkJDn3fbZUikUVkS/u6igNKpUwM7xmaxTp0nrlZGDb6raOUZA3guQs12jO7So4j/e+qCCj4LmTQNAIUHKChjpsDhRSyTl4UJNgt2wxOTZnz541C/dhLZzSpUubUD8W9sOwFFY73r17t1kjB9tAZPDPWvUY+ToQIOwLBdGc+Ph4sx+7xemSg/P454eLZYafJKeeJiP3bVFVbtC3lJfSYakolUYrgIZIzr80aoTSrlYZqV0u1iQrb9h/0gxvTdQnJMZyvEdylmw/aqI4T/VpZLcbcjsSIIE8EKDk5AFePv1oSCXHSUzDRXJ+XntAYiILC1682VpfyzBDp5afPJPiM8qikUWklspLi+qlTcSmW8OK0qhKSRWeSHGVHOwY77hKxdBiUqrULBsrE//RVkoVizSvhGC+jmf0iORQcnzulvwACeSJACUnT/jy5YcpOenNGg6S897vW2X1nuMSGxUhVeOKSqe65eTuj5bI7mOJee6cVzWtLHd1qS1Nq5aWH1ZeiOS47ziuWJRGf6pJudhouUKnpUOUXEuK5u6c1ZlFyNuJiriQsIzhtmOJyRKhvy8RY+9VG3k+qRx2cDo5xdQFkhaI2U85SU7i2VST3xQdUcTUgYUEQkEAMwCRa1dUH3KcXnD9OJt6Tg6fTpKyxaPNC4Y9FUqO01sy+PWj5ISR5OBLjptjER1XitAvORYK7D5qjmzVoSR/lEd7NpBLVJzmbT4sL09fl+MuI4sU0gUGW0pvlSPXgiGtfcfPmFlFWJ/HKni7+gJ9MWjx6CLmVRKBEB3wSZu6XchEp7IruLDP33JIikZGSI2yxXQmRrQ/8GXah6vkPHlVI3MzQYH8rdh9zLxtPr5MMakT5Gmufj9R7jAsCSSpaG8+eFowk/KiyiWktD68OLlg4sU2vc7N33LYrPmF64unBwRKjpNbMTR1o+SEkeS4dxF/Sw4uGnhiSrX5nqw3B7WUPs0yvxvsPs3l+U2ncN7SsaY80qN+RpW/X7FHXw66xQjF7Z1qyiX1yvu9x+88mqAXwgQTQWqvuUSeCiTxVFKK3Khvdoco3nZxTbmuZVW/18VVciCPB08mmWNU0Onk//lutRlmvFIF8blrm/j92DntEE17Xv+nWW1cBymo5J11sI36MDL+t62ycvdxuVsjuFg3KxARTX+d9fp9J/XVNhs1yrxPnu/XVLpdVFHKxmYVM0qOv4jnn/1Qcig5ue7NniSnzxu/mzen/+uyupkk5+MF2+Uj/Yfhrls71sgSAcp1JVw+OGLqapmmSdOIRo0a2MLjLhFRQRTl1gl/yl6d4j1Yp9EP6VrXH4fP2Mc5Pcb1b8+Xv3akJR73bFzJLOSIMu6WNvLZnzsyprkivymYBVG2XSqDWLsHT8NcDiCY9J1zrIUaEXl/3jZZoy8A7t+qqllTxsl9YcWuY/LfH9bK4m1HZbgm8vdqUsnj+lOUHOf0MafUhJJDycl1X4TkIJdn++EEc0O/VNemQIQE79e6Ty+arpGcD//YJh8vTJOcWzrUlD7N04a5Pl+0Q/7celSHtmLlyiaVpaZbjo8vlXv4i2Uy5a/d+pRXQd77u2d5gOQc0RVR//7+IkHkZ/ClteX+bnlbNOyAytIZHSYrpblGsToch+HDByYtMzcQSE7XBuXllvf+NKfy5T0Xy1dLdhnRuUSn4398R3tfTtHnbTFLcC1dwgAAIABJREFUEUMSGMJDsjkWS3snPaL2yg3NTaJ5fiiIzuFGiBs31iZqWrWUDommraPlqYDJ4VPJKnoxJjeqoC1K/cfmQzIxXXKu1VmWD3Wv72jJwXIMz36/RpbuOGZmK15JyckPX9ugnAMlh5KT644GyUHUBKswf/bnTnMzx00GQzOQnIe61zM5Mok6nv7Nst3mxu4uOXd+sFgW6FNlhZLRMqB1vNzTtU6O9UHODyIxFXWBQsz0ci2BkBzk+UDazmgOA3JoPOXv4LwWbzsijauUkps61NBZVUfk6W9Xy0b9nK+Ss+NIgrlZYybbtTqMFu2SvJ2bhjqtN//RP28UJDtfrWK5Yf8peeqbVRKnof6ZD3UOSD5SdvXcownyWKgN54ThEfD0V0F06rvle2TUTH3Zr/ah61tXy3b/ydqm6HPTlu/VxTBLmaUUshOiRdquxxLOmn011PWi8ksJS8mZppKjspPXSE6gXlnDFY+d+e2g5ISx5CAq8cqM9UYqVukQ0VqNHASz3Hd5XU1aLCm/60tDv9YISlO9YWCMH1IDybm5Q3VZuOWImf0FSfhx1d4sknPla79l1LuHJhS+c2ubHE9h9M8bjHRgyvt1Gma3CnJtHvliuVm00G4kB2v/3NAmXofW6kh5nZKP94NhtlW0RjyspEbc4HDMk2fOmjWF8PZ29+IaIof4vTV7s4nWHNB28VVy8Lb5d3/bojNJzsuHd7STkl5moiEigbpVLuV56Anv8rni1TmC/3+mbxMTsfCH5ECetquQIVeii0bwEBHCcAcYor1P6t9b6JpKrjPs8BR+07sLdAmEIjLmplbSsXZWlrntv5DfSSqbH87fbnKsBneuLfUrepaSvcfTZOutOZulriZ+j9c+h6E7T+UlTcBfpX0a53inRv1yWyAVENfa5WMzks0hnvgOY6aQK6fcHsOXz1mSs0iHf9rUjJN+raqZB5acol++7N/XbY8lJMtBjbAW1lyxOi4TFqz9IJLzjEoOXpfiLjlguFeHYbG0xnPfr5Vv9BrgacVjPLDs0tfVLNN+iKUz0OZ5fYhwPU9Kjq+tHpztKTlhLDmoOoYj8GTyyswNmpi3KTi9Jv0oWFAQ6/UcVRHAEA2GCFbqFHdIzgCVh+YqPRN02jtk4sZ28WbdGPdIjq+Sg5wf3HSQ1/PQFfVNRAIFN93H9FUUU/VpHhGlsTe19jjDynW4CsNJLfVid7NGXxD+xmrSuOjhVRfWTQ/RB0gCZne8fH0zI0U5Sc5L/ZtJD53xZq1d5KvkvD9vq4yYusZwnfFQF531kv1wEjgjagSxbKtrJjVW/jFuQy+Bkhwkrk5evEs+1eHG9/7exiw9gBltuAl9uWSnvtIiScACNx7MK8ONfMWu44KEbBQsuY8bkbeCvo1pw4gIYr0mrM3kqaA+n6m4YFgUEbB/5iA56IfIEYMQo73BGbPsPJXBugAn+kXfFlVk9MCWPg1roa9BQtGHcPPFsCZE+e4udQwTiMZx/e7g2NXLpEUl/S0ZOCaOBQmFXFrFkhwsLoqZkpDkCbe1zTQj0lvb4O9olwMnz2ik8JzE67IWrsew83ksOYHI7GLtx9b1Afl8VsGrUSAnuL4gIon+9eAV9Uz0rXp6JPCUrhP2iQ6Fo3+gTf/UWZyeJAffhR91eYyXpq/XHKS6pp/ge+avQsnxF0n/7oeSk84zHNbJyanpR2pE581ZwZUc1/rg4tZYozoQBzyhuhcMH+AChRkRrjk5niQHN4czKalG3kxkxWXIxpKc1jXijJxYM6O2HT4tz+mTHhZLxN+evrqRNNGbvvuMEXfJQT0va1BBntVZTpATyMz/VGYgaSh4USmm6ftNcu7WnJy/POfk4HwhOXhixcX3pwc7m5sGIiTg4J43sk5ZI3LxqUYwyijXT+5sb6IErlGBQEgObjoYokTECpGcEdc0kcs1Dwpti2gKZBM3nBk6HLZD87VSdCFJ3MRxw/JVcnCsj/QckTB9WcMK0rFO5ugP2hNcNmt0b5JKzgcqOdeokNzVuY55HQkKbvLYDksSoT/8oZFHJMH/uGpfhuTgBo1Zhchdcp2abEkOcs/+N6CZilz2eT7ufR6CM12jlyNnbMiYXYdhujdubGmWgugxaq6JSiLxF/loODaikP4qSIDHjMYk/S5119lIkGBPkmP97scHLjWRWV/Kdv3evauztBBJQ6QwU/RMeVurQIE/6oEZjZG6kj3aDIJzSKOdGGb8UiOf2Ae+i++7JOOjj+04ctoMq87bdFiW61AuIoD/UkmxIoGIzHV/da6ZNWkVSM4HbiuzQ5Sm6HdvvEZKO9UpZ1634ilq5Mv5u25LycktucB+jpJDyfFLD8NFCxevs3pD8zQDHU/5eOprpQJyi8oJLvYo7pLzxqBWeuM8Ic9OW2sSg2/VqeiY4mrJiiU5uJFj1hJuGCi42OIzSKxF7sQ9+rTcR4/hPmPEF8lJSc/H6afRB/9JTsd0ydmZJfEYQjJRb9Kv/7JRFzxTadGk5Fc1QtdBL+qIJGARNNeC3B1IDm4QsRrhwEyxgRopwFO5VQIhORiKQxL56t0n9KZVSP5zTWMzfIg6u0vOMxqVWrj1sJFR3Mh9lRxwv3bMPBMNGdqjgdymyw+4ljV7jxvxQCQPSefICaujSez/vbapeSUJCqJqS3ceNa8nKV8iRpZono2r5PykMnZGj4OIY5xGzsDbekmwJTmItvRvWU3u1yiC3QLxflAT0DGMnKR9H8WT5OC7AZnFUNaXKsH+KhjGxIt2kX+E9awgIVZxjeRYv8uN5GBoCH0BESm8pw2RTsgGvjtYZBBtg3PHIqYv/rhOrtV+DAZY0BTRvjdmbZQ5Gw7Kfo0GYUkMd8lB/b/VfD5EgDEjEJIDCX1Bp5FjiA3Fk+RcrDL8js5kxPfCejig5PirZ4XXfig56e3FSE5wOi5mT13fuqrmqtQ2w05DJy83s7NQkJODG+b9+nLQ1ZpjhIgQhh1wc7OiE5bkYPsr9On0XR0qQcE+njWRnP1mFeab2lfXG2KtLAuGeZKcyprEDPn6SZ/s8XRpRXIQpUB0DLlE+L2n4aq1Ws//aRQNb4fHIof3aqj9fp0ujpszivtw1ZsqZdM178ZMdXebXYVZTxhGQfIxhi0wHIf/RoQBsuf+NnNc8BHlgOTgRnmLDuEhSlZNoxKYObRHn3Bxg7/n4yVm2MQ1JwfDYF8P6WRC/namDiMakpweXRul4gVJwDniuP+5WiUH65Z4kJxHtX2xbhLkCzdAS3LeubW1uaF5ey0IVqbGUzokB0/e6A8oEGlEhpAvhbbB0ClyoLAqOGaRQX676v6xyvZ8XdxynObfYAgSuWK4+X+o9Z+u7Y3jP9O3kSzYfETWaSShpbYhokC1VDhQLMlBNBERwk8Hd8j0RQFX5Isg+fyxKxsq+ws8cVPF++bQN5HsjOJJcqwd4sbfWvNjrmpaycz4QrIzIiOehrDw/Ziqa0811lex/KLRyys0AoQV0CEDVjmueS7/p2/lhkQgEvXKDReWVfAkOZ//s4MREiwMiEgiIisYBo3WIcfs1tDBmk+ITOJ71VzZIb/pvzo0h3OHmN/YrrpJ3B7/22b5Qoc38WJgRKv2Hjsjny/eKVt1QcKjWk+0IQpyn96+pbX23yStQ4w88fVKmaz9G/0VkrpchzxxLOu7iJ8hk33fnJcpkoOIJh6mIOE9G1c0covFBFFXRnKCc613ylEoOZScoPZF3BQ760KAN2lS8gd/bDdPmVaYGRc4jJUPGr/QjMObm4wme2IqujXWn53kQAwgOcifKKmC0F6fxsdoVAhyhJA4RMHcLM4X0uOdldt0CjmG1lBwIcSNEWP7uNTiKRHTaufpsMbQL5ebp1T8/t+9GppXWSC/xMrhwAUd7/qCUGC4Bu//wk3Vumg30RlX1coUNTdUFLzFfeeRRHNhdpecx/WC/unCHWY71Lt19ThTR9xI/66LFo5QAcQQBG7myD1CLg6iBJjF5S45s9cfNDdePFH/pueBKNpTuvIynmoxHIab9lXNKpsIC3KQIAw4ViMdrqitNwj31WQhnG/N3mRY/KmRkLV7T5p6ol3+rnKF4T0I7Jd643IdrspOcrrp0BOe/HFjxo0K4jF3wyGTI+W6fk92koPhHswUe1KZIVqFzxxTThBnFOwfQxboR8g7QZ4GGI3VhGe0nyU5uHm3iC+l7ZFg9oMIwCMq1RAa0//Sc3LwM9j8oEM6aFvkuoA9RBhDTj9o++Lm3EY/Z+UNIU/ohnHz5YT2K2vF65wkx/oiYsZXhEZFa5YrJj0bVZKemi/mXiDzw3UGH6bAoy166Y18UPsaGfXG9kjmheT8plyxrgwWngQPrDaO7x2ihvi+WOV2fSiATKKv47sCOemtwgWBxTlZERoroojz/q9O6/5V+xpKvAreV0Multt0DSr0JcyUxHcGwv3i9LXmWL01mgcZ/l3FFzOl3AsE/XJtOxzrZj2f93T4Ft8vSA4eXpDX5So5+B5B9CHd6ONWQX2ra/QNw2U4b+SMYQYehqswhId+9wSHq7Lwz4+/oOSktyojOcHr3hAEPCFieAE3K6vg6cvc+BG9SH+yQ3TiQV3HxrqwukpOK5UAay2ejQdO6g12l5llhoIIBXJCsB8IRlp0BImrKlmYKaNT1y3JcT9zJDXjIoibGJIdrYLP4WkS9cdaLCjXjZ1n1u7ITQGD1/7Wwszswk3tiW8uSA5kBKF+zErDOViSg2EZDMkgDwc3JKtAipCjgKfkLvUrmKf34d+uylStAZoXhYInY6vg5oebGyRngt5Q2tQoY9YNco8eQCCuHD3X5MZcaLG011QgmoCZOYhkIe9qrOZRWDk5luT012P3U6G66d2FGcd+4qqL5E6N6J1QwUGeDG66GCqqrDduDLkhkR37x5AmzhWJong6x+8hpohmYKr+Pq07pAxt7FrqaDQGXCwhw9+QJA2htYar3NsNK2VjzRjUA8VVclCn+7vVNX0Ws3lW6g0Xw6SYCYc6vKrrDqGPWNEU9B/U3bVg2AhRiGMqVDfr2km4SWdXsB9I6HDNL3MtEJWpeuN/+PNlGb9up30J0oj2tIolOb+uOyhNtI0QWUnQ/jSwTXVZp0PC7pIDruB8WOuGAlHAKxSG6ordO83SBsdNXg2EEt9VSAsExDoHDCeBAWbvYejPVXIwS23mmn1av8pGaBHJ9FYg9YhSzd14UEprjheia5bkIMcP/xCpxOxGRIOyW6h9vEYN29YsaxKk8XBgzap7VSNbeCjxV2FOjr9I+nc/lJx0nuEuOcjjwOwqa+wfT9WYUYEnTlyQnVQQRUBo/ohemFwLnrqQC+EqPrhp4DUQGHpAcZWcMnoTaKjv3cFrCvBUjyiDFUGpqhfiyXd31PfznJKxv24277xBQQIjkpJf0WTQ7G4wuIjXr1jcPAVaT6mu9UTE6T2diYKbCCJCVvTAV8ZY5wcS10FvrIieYLEzK5Ljvq9BGvb/tw6HQDIQHUB0x1OBMGFYBxd/d8nBsBzkaY8OFbiWEX0bm/ehPYn1c5Qp5BD7sQpyJXYdS5Dr35ovh3QYIbuCGymkBEMKmPqOHA/cgLDEAOQVyZ642VllmJ7PHRqpw431NR12QiIwPg9pQNRkkA45os5YWBHHLa7RhS66Hwx1YgqwN8nxVE9IDqJPyGXCdH33gkgMVv/FDD0UV8mxtkUd0UdxU4VcQaTwvXt1QHPzuhIM70BMIQbor64FYnvbxTVUyBLNUgGWUHiqK6JMfTTa9np63pm1DdoeuTBP61CRVVwlx3p1B7b795crTCQr7X13hUw90S4Yivp4wQ7BOkBWAXPIGiJaKLEa+bq0PiIejeRNvbZg5hKiJchzwfDYH5oIjPpb31eIMV7zAq7YByQHQ82Ytv2irlaMd+J1Ua4QdQzveisY6kLeGdoZ+8ZQICJn1vHwPcRwHoafcipYr6tehRKm72M19jfSJ2hgYU60t78KJcdfJP27H0pOOs9wl5zJOkzwgV64cWHFxXGE5hlcWre8Pp1vM8l94VyQe4McDnPRT59C7u18ymlIHDd7DNkgYoM1SnwpuOBXKhWtF+O0ULxrwQyfl/o3FQwJIY8GF/HcFtxwuzUsL/dp9AQRGqzf4qngiRqRGNycxmikBNPHsyvPaoIpxA/DGa4FqzGjuLPwJDmQHTy1I6B2XPNaMGsNazJZU+O9nS9uSpM0xwNDY1grCWKAF0cnp1yIAz2oOUeIzOAJGzd8SI6Vy4T9gw2GLyCjEC2USN0PhpI+0plk6/SGh0gOEk8hGztttMObOoSJukAmf9MIgXtB0jrkqq/mjhRXUbtb85lch3TctzeSoyeWpNGVkTr7qrkO/aE/gB2iWQPHLfCGKse/Y0gRw66uBVE35Ja8rFOhrYKoJt7DhuHUtOifLsKp07qRM+MuFN/d28kM2SAK41rcJQfDe5dqhO5xjbghAf7bZXsybY9zR/9wfShx3QCSg7WF5mpU8SVNOoao4qW9ECVEerwVPKThXCBmyJ2CXCHPyDWS6G0f+DvkBuKHYXI8FOE7i0LJsUMv/Leh5KS3YbhLDnImcIOwhnmQl4InTKwAG+6SgzyWkdc3lwr6VH+1TcnBlGoMO4zXCxpmadl852jGNxphbEQREC53L+CKSAUu1sgZsZjn5nKAaBEWH8SsEzyxIs/GU8HwG/IvVumMJhwzp/PBYmkoEAzXYs0ycf+sJ8mBTH2iIrBb2SH6hTwc5EnY5YiZNYiYIe/KWqQS5+p6g6qkwxsQL0QZMN0fCdmukoPtISRWdM46lwb69I7p3Mg3wvoziByh2GkH5LvgGBjacs3hsPaNaEcxfTs9JO1jFSns/xfNf8mpgCsEHLlcizT3B/k/mJoMOUeeVV4KEoaRRF0Y89/TC6IRGK7EsK5VIAGIrkAKISPgjCGlLRrJRA6Xa8GrRD6YnzkfB3/HsCMihVZ0CW2IyBNm7UHA7URfXI8DycEQ6n06kQByDBmK0e8OMmc8sXfnhFNOi0qlTUV3j/L6yhXfW8wAxfcHhZLjK8Hw3J6Sk95u4S452XU/DMuEu+RgPB5PgLhI4YaGcLe3YiRHp/viiTU3kRZc8JGDgQTPYBSIRKJefE9nE6HBEB9uZK5rgWRXLyRu4sbwq40hAewDYX/cpLH2DZ6eMY0ZQ5/4bzxJ49hInvWlYH9InMVN01VcXPcBgbFuXpi6jZurHVFBhAHDD7V0/wtUKHAMuwX5OEhC93aTBesJt7WRtzVqhmEWbwXnAbHBwnYYJmqpkRXkxyAZPi8FSbrIk8LwGc4bQ88YssEQKRLOXQuib/iOINKDYpZtUBG0omDWtngpLIaq3Ps2IiXgYiX9ow1LFo2QKvo9wHGR9+JLuUuXfkDU7Z8fLsn4mKVqvkZjfDmu3W0RIcPwWXYLTNrdj7Udh6t8JRac7Sk56Zzzq+T8oCt8YoE0JOx5WqQvON0s+EfJq+Tg87joY/gv3ApmeaHklO/hfnPEf+MmZq0/hLdUu0cAnMQB0ROcJyIEVh6aP+uHiM4InRqPJG0kU/taIOWYju6a5O3rPqztIZn1NEcMw2LHE1PMtHgMOUJAc1P+pRGWuTq7KbuoYW726ekzSN5GZAlLHDixIOdnQJtqZvkBCB2+73kplJy80AvcZyk5+Vxy8B4hvDwSeQV2ZjQErqsFd89IbMZMI8zMwFRYFhLwhQBueug/iJjkJnEf+TjlVMLs5J74Ui9/bIu8IyzE52tkxtdjY8gP0owp5E4sGALGGjq1yhU3K4a7zkzLTX0pObmhFvjPUHLyueTg9BCyxisAhk5eYRbZwtCDFZIOfBfjEUiABEjAmQQqlIw273vD9Hm8NDYvhZKTF3qB+ywlpwBIDk4RyXZ/7Tgq05bvlTk6KwhTRVlIgARIoKATQJ4WonaYeZeXQsnJC73AfZaSU0AkB7MU8F4prCXzf7p2BpJKWUiABEigoBPAW9i76gKa49NfEZNbHpSc3JIL7OcoOQVEcqxuhJkuD+i7lXJa+yOwXY57JwESIAHnEEAS+6W6hs6H+tbyvBRKTl7oBe6zlJwCJjnIz7n306Uyw8Nqr4HrZtwzCZAACTiXgPt75HJTU0pObqgF/jOUnAImORi2embaarNgmLV8e+C7GY9AAiRAAs4lQMlxbtvktWaUnAImOThdrJSK1x1ghV0WEiABEijoBCg5+bcHUHIKoOR8AslZtMO8IoCFBEiABAo6AUpO/u0BlBxKTv7t3TwzEiABErBBgJJjA1KYbhJSyVm6dKlMnjxZVq9eLSVLlpRhw4ZJo0ZpLxd0LQcOHJCpU6fKzz//bF7Sdv3118uVV14pRYsWNZv98ssvMn78eImJiTF/L1KkiAwcOFC6d+9uu1ny62sdPAFgJMd2t+CGJEACBYAAJSf/NnJIJWft2rXy66+/yr59++SHH36QMWPGSPv27bPQPnjwoCxcuFDmzJkje/bskY4dO8ptt90mxYsXN9tOmjTJfPaOO+6Q2NhYIznNmjWTunXr2m45So5tVNyQBEiABPIVAUpOvmrOTCcTUsk5ceKE7N+/X44cOSKPPPKIvPLKKx4lJzExUU6ePCmI/CxYsEDKli2bRXI+++wzmTBhgsTFxUnhwoV9brECJTkLkXi807yXB+/owduK8R4XvBL6gK6EfMih75rxuVH5ARIgARKwQYCSYwNSmG4SUskBszNnzsjGjRtlyJAhMnLkSI+SY7FdsmSJifhAZNwjOe+++678+9//NtGdChUqSMWKFTMiPZ7aJjk52QgW/qWmppphM0SBRowYEaZNab/an7hIDt5sjDcFD2pf3bzRefaGA7J853H7O+OWJEACJBDmBCg5Yd6AOVQ/X0jON998Iy+//LLJ6zmnry5o27at9O3bV1q1aiUREREeT//QoUMybdo08y8hIcHITrdu3cx+8ntxlRy8t6VJlZIy6m8tZOmOY/KRzrz6c+uR/I6A50cCJEACGQQoOfm3M+QLyTl+/LgcPXrUDGNh6OvZZ5+V2rVryw033JBtXs55XRXv7Nmz5h9+Hjt2rGBY7Omnn86/rZ1+Zu6S07RqSXl1YAt9U/lJ+eCPbfLbpkP5ngFPkARIgAQsApSc/NsX8oXkIHqDIafIyEhJSUmRcePGybFjx+SSSy6RLl262Gq9gpSTs2DLYZnw+1aZsWa/IJLTtGopIznIzXlz1iaZMG+rLWbciARIgATyAwFKTn5oRc/nEPaSA6mB4GDqeFRUlCDX5sUXXzT/3bNnT2nXzt5L1wqS5OB1DvM0WjPlr92yYOthaaaS88oNLaRq6RgZ9fNGef2Xjdn2+PIlouXey+rKrHUHZMUujaAlJOffbwfPjARIoEAQoOTk32YOqeTs2LHDzJbCOjmffvqpWf/m8ssvl0qVKpkhJEwFr1+/vpGYP//8U3766SdZtWqVSRDGNPKrr77aJCEjcXn58uUmkgPJwc9t2rQxkoN92SkFSXLAY9fRBJm8eJe8qxGddrXKyH+vbSJVShdVydkgr6noWKVD7TKy9dBp2X8iSWKjI6RtzTgZd0sb/exO+Wj+dlm//6QdvLa3wTEub1BB5m48KMcTz9r+HDckARLwHwHMtmxTI06+0XfcncML7/J5oeTk3wYOqeRgSviUKVNk06ZNcurUqQx5gdhgCAqRmc6dOxtx+fzzz81igPgZU8Sjo6PloYceMrk3ECUkEKNAiJo3b26SiBs0aGC75Qqa5CCa8/Pa/fKFykqX+hXkJp1dVSY2SqM7u0xeztbDCXJCJWN4n0by/cq9smT7UWlQqYTcdnFNubFddZm/+bC8OnO9LNp2NFvGEBbsc+eRBLONBtckNipCInVG18mks5KSmvXiWVVner3/97by9NTVsmznMUlMTs2xDYsULiSli0XK8QTd3znnX4wrlYyRU8kpckpnstkt4IVyVt8gz0ICwSDQ7aIK8miPhnL1m7+Hbb+L0GtDtbhisk+XxThzNufrCCUnGL0qNMcIqeSE5pQ9H7WgSQ4o7DmWKBs0EtO2Zhkpqvk4WDMHorBQc3YwbLV42xF5+upGMm1FmuT0blpZxt7UygBcpDOw/jdjfY4zsRrrrK1eTSrJ27M3S+LZc1IyJkKaVCsl5YtHy0pdo2fzgVPiriU1y8XKzIc6y/vztsnn+n6tLRpFKqT/y+5pEiLVXS/IszekR350h05WnVs61DDnDoGzW2opE7TP/uNn5LDKaW5KceUESUpKKTiihP4MsU4NA/nNTZsG8jOQnP/r2VD6vBG+klNCrzd3XlLbPMjt1mtdToWSE8jeFNp9U3LS+RdEyUEUGvJQWJ949F6QURap3LwyY4MgQTk7ycHN9t9frZCfNXk5u9KjcUX9fGN54Ye1Mkcl5Ob2NaR/62pmXR7IyzX6lOgezbEkB/sdN3eLkbAIjWQgquSpVNAcoRf6NZNnp63RJ7ZEHeYUR9/IX/9bS0343mfE0W5BNA3cED27++Mldj+WabvBl9Y2coU2LSilnMo0In379UmexTcCV0ByejWUq14PX8nBQ9U/O9eRSfqwtOsoJce3HpB/tqbkFGDJya4bI4LzikZp5m85Iv9Jj+QsdovkIOKDYagP52+TH3Q4q6gOQ21TcbHKgDbVZEiXuoLhp80HT8kvOjTWsU5ZaVK1tERHFDZ5Pt1HzclWcn7VxOa3NAKEnJ8IHZ48ceaC5FQuFWM+d/BUksRrOPqbezvJQb2RIYdnytLdGgHa6fHUmmiC9b2X1ZFSRaPkyW9War0u1DdYX+nP/tlB9ugF99M/d5jomJ2CfKlBKojzNh+Sm99dmOkjyJ3AENif2mY5lXduaa2J5UXliyVpw5H5qVQpVVQ61S0rNTTiNfKn9RmnhiR5DFns1QiYa0HfbBkfJ7PXHzAzDPNreaRHff3eHfApamixuOKiivLvKxtI79dylhxEak/qd3PeJufJc6mikfJozwZazIoJAAAgAElEQVTy7m9bZdvhnL/rjOTk12+Bpklogq+To/tBI18QIzl2JGfENY1l6vI94i451mcRaYHEIMfnyW9WmUgKyn2X15WHu2tOlIaIUnSYBENjyM8pERNp/n5Mc2jwhIWp7AdOJpnfYQp7y+qlZeI/2slslZyxczbLPr1BYahlow5tWaWdDq8l6Bg7XktRo2wxmfFQFyNOyHMZN3ezvKHT4N0LxuavaV5ZhujMMGzb760/ZKXODrMKftekSimJ1ae/jXpO1o0Rw2HXtawqP63eJwfT65mXTvnlPRfLWR0yghz+uGqfiaBFRxaW5JTzHofk2ui5gmWX+uXNjLib3CSntSaHNqhYQiarvGB4EEMzu1Wijuist0oqg1fpECMSyjHUiGFCLPY4QvOdrFJWf4d8hdNJ9nOEvJ1/lLJEFAURlGY6PHlYXxOyIz0vy9tnc/P3uhWKy/WtqsmVeo6QabQ/Zv3V1L6BfCbXvoP94+aPvLLtmne2ft8J+Urz0NbtO2WbAXLAEpJSJdlmjhTqgHbB22YQM3XPrUK7YWhlgT5U+LN8cVdHeU+/X+i7ORW0V62ysUYEMJwJGUafv1b/Xfnabznm5Dx51UVyRj8zUYeXD+lDhy8lrliUPHBFPdP/puk1ZtWeEx4/DlFFtDnZh6HWsnqtgbgg13D4t6vNAxBGLT1F9epVLC43tI6XwZ1r+1L9LNt+8cUXMmvWLHn77bfztB9+2L8EKDnpPCk5FzoW8kXGzNooM/Up8MV+TeVrjY4s1Bwc15wc9264SSWkx6i5GTfq+y6vZ24m2RVc9I/phefGd+brkFSawODJC0/kr/+tlfyqT9lvzd5kXjWBm4prgvPANvHms7h4u0pOkl4sP9QZX2M1AnRak3txUYzTzyIfCPL0j4trCYbQUDBUhinwGTd7vSjigl1Sozzv/7FVft+YtiAiLpbPaSQF+UdbNPKDyAl+t9SHnBpXBpAciM2HGk35Vi/suAE2rFRcapUrbnKJIFjr9p00+VHlikfJv1TKejSqKJX1xoNIGSQFsgXhwZDhZQ3KSysVHZzz39pW18hWUcMF7dW8Wmld/6i5VCgRYwQSQzf4/FMqo5C6Ho0rSX0VpD90Xxii9EfiNvhAbMB+rjK8Up/0cfPEatp2CuQoKSXVtLtrQW4NAHl6ImukkoCbGcTltMrHmF83mShjdZUcpGz/rue3PL2tMQxzW6dackndcmb36DPDvl6pUZ2DRtS9FbTL9Tp0uFCXXoAkWQX166eihSio+++7a1QE/RXtBrl3X3bhmuZVpKJG48b/tsXb4S8cT3/CMV3TjWK1jTG0a81K/OnBzvKmssBDSk4F37t+KjToj6hfT/2O3HFJLYnTft5rdM6SM0rX14Ks4GEFMyLtFnxvB7aNNxMZkvT7+bG+auY7PT7k8aLKJY1Y/bXjqEDAIDnoDwleJiG4Hht5bNg3rlkYGt53PFFW6EMRhnzdC/ro/d3qmePmpVBy8kIvcJ+l5KSzpeRc6GR46v5BLww/6s3y0Z71ZfzcrSanxp+SYx0NT4pr96Y9weFie7EOab1xYyvzDq2xv6qsaIQBF1vXXJLHrmxoRGG61g9PnbjQQhYQRcKQzhSNaizbdcwMieFmhhtBy/jS5iZUu3ysOZa75CBPCBEhyMCjk5ebyAgKhOYFFb2Xpq8zLy69qlllEznBzQNPrr7GQSE5uPF+MF8lR6fnYjFGXGRv0oTk4tFFNElyl3n6Rj0G6M30Dk2cjFdxwA0Nx0rW2YPbDyXIw1pHRLL6tqhiojWTdIgON99W1eNktK5zNEmHwzrULiuTdHjMtViSA9Yf3dHOzJjDUgCQQ39EW3Aut3SsqdKwXz5ZuFN6qUhBcrIbmkM7ltS6YChxvcrdpfr0jXZzT8yGoCHoDMmBaLgmE0NcbtUbWud65Q0jtMuBk2dMf4JsoZ+8PH29uWEiJwrbQwZQcDN98cd1ZgYhooYo2H+MRtcSVYDc2zeiSCF5R5dQeFeF5A+XGybq9/39l8hoTdifqUNgVv2wr3dubW1mCeI7NF9zohBpcy1/axdv+jHy4OwW9Ad3DujjdSrEGsnC+b3cv5m8rMN3kBywQGQLooB2dk1AR8Tvheuaat03mPXFIIz4rqCet77/p/muQdBOeYj24buHqeZ4EHp15oX6Q9pSdYZsdi/7vbxhBZlwW9uM012nETU8WBxVyUIfQA7Na9qPS2mEC0n3O48kmllS5jupIlxGH15QH/ehSPwdooyh8RuVa7taZY0kbdFo80R9sEA/dy+QY1xTwCgvhZKTF3qB+ywlJ50tJedCJ0MyMoZ+Vqt8IBrw6JfL5XuVnmBKzhyVnDEanUA9ysRGZgrl48KKIau9+nQGAaqtURDzpJ9eEHHBqysgEfd3q2ukpJqKAm4kVnGVHNws8OQ37b5LzAXVk+S8rxdIPLH2b1XVSAdWhsZwE4bi7Iz3onq48Xyu0oEbJfJivtLFGHG8Piop92rkC0+37+tq0yOmrtHoS7T8+MCleu7Rmc7Nqv8N4+abmW23qlA83ruhDgklmeGpIjom8tz3a+Vjjdi00Jueu+RgaOZ/esOHWOBvGEKELGEGCvKZXKe2I9pTTHOtMMxy9DTC/bp2lZGNtIR1FJwXzgE3TdzY/6Y3DAyv4TzwpntID6TFVXLwUtjievPCMAWiR231Jok2Hak3edxwfly118yucx3WwY0Kx4TMQgwxVAQJwX8/3L2+/EOjM6ive8FNHYJ8xweLzHAdVva2ojjYFnX+fsUeXeV7m6zW4RIcE/vB8CZkwH1oKVLbbuLt7dIiFyotZ9OXQQCXnx/uYnKtpq/cZ0QLdQTnKSq26F8QzHE6BOueBJud5ODcdLceE+k9Sc79yv0OTTDH0DCSbiuoaDz8xXIjOe11LSxEK9Ly6LZr4v8p5Z82066mDlVNva+TOaeWKsmIDGKI+KhGtj7UOpfQnyFoGPYDrwSNklqRNrQbhrawHMWdHyzOwI9oa6q2Fz6HSA3ydly/J+6S49puaDPkS+HBAlFcRBuxRISVP9W3RVXppBKD9vpS+7P7UCv63JCudc3wLYa5UCDIeHjAd4OSEzihcOKeKTmUHI/9EhckXBggAP/69K+gS87cjSo5GsnBxRFygeEXq+DCeq1e6PBcjxwHV8HBNiYpWZ88cbPs3bSSlNfhGms6sSfJwf676LDPS/rk+//t3Qe4JEXVBuCWIBlWJEvOQXIGSZKzRMk5I1kyooAkEURRQKJkUILkJDnnvCAgApKTLEGSwF9v7Tb/OMzduXfuLDs9e+p59mHZ292366tTdb76zqnTHFw9yTksKTkWf46D4+HAJFsfmciC3arUjGYF0xCb6SccOzvZAcnxceLCE5zJ1otNk6T7KTNJqSU5lCWLfKNWS3IOXn22TDxKHK5O74bkaWdv9b9KjpAMRyckpE9lvsjA5DAQHY64bPJ9Vp1jshzyO/Cyx7NjgyVVokwER1iWT04F4ZLvkElOCrH5NAh1aaUhJEdOl4YMUI82XXiq4ra0c+eoEMcdl5qh+Cxhicz8Od1n1y1sVzb46SPsZkhODwnkWKdLmG6/5HSZgDdq3hXJ2SKRHOrC0evMmUKig0NVZUNUkU5J2dSkoZOckYprd10sq0MXpusR6rJfSA5F8LYUtjnn7hdziGyT1M+N0x82RjHLJKfuOPN6KWwjb6pWCfFMzl2uTiMVjO8uSz6U/UBidkuEz7wt58VO5z2USc7ySVE7YaN5sqr5SAq1StCmKmkI2HWpbIPmmcZIswZ8zrjTvyE8SCnlBCk8+54XM+HpieQcBedERIyhpHjKV+1Jyr6QnA1T0v24KZevPFl44XaL5PDzI0mtPSe9x0VDVNcSB6G2/VPoWV9qG5LjFGZ9CyWn4dTpmn8MkjNkKEPJ6dmm+0pyZkyJfI4sr5N2c81aT+Gq2559M6sljo4PSGqDujxlKxfWnp49eHH+shiUCNJ4aXHkIOtbrZJDmeFotkmJh5x+I5KzcAr92JVrdvdvJIcu1o/cUHTK/B4OUphIuKXMjeDUd05KzfQJl6nS7+JEOAt1guRqLJhUqQnHHS3/7t6SnPtTEUZky7vbsdY2mCEzcODAahvH5P05O4pI6Qc4MAoAQiB8814iNHb/G6TQhTDQs0nut/PnUO5MIUHJ4sI/U08wZnHMOnPlk25yhuTkUAJufurN4qArnsh1SvhMOVuvp3vkrCAl8lAoMZSjKZOS4MRc2ThUBMlOvmwTJ3zkVo2aHub5S6dwh1CM03We55RPo2Z8XkjhMuUIKDjGRjijvl2SVDV5IQjF6AmXaVK/nL6rT3Y1RghBVjpScjN1wSlAyqKwJqKK5FBL2IC8EOGfAT2QnL1WmCm/k1BNPclhc5Q2oTQNtkiTE4fGwc+oGj7P4l2QHKpWbStJjrDhiemEnYb4UfSQWnbLRq5PhLrRPCmfxW6Qp0/TvHoqKbzrnXR3VqrKuShEpq7VSQlnieD7rDhLxvvjRKaFijY4+Z6kuo6ac36Q5b6QnE2TWkmx3fz0+7K9XrLDolml1A9zYGB6n10vePirwqFbpU3DASvP+rUxDpLT04rZ3f8eJGfI+AbJ6dnQ+0pydkmLrWO6JP9mrZbkCNesm4jRdktMlx0FkoMEUBJqE4+bkZxmv9PPS5LjkxaUBLks06XF2e5PfoFwD4cnJ4eSI8ZvN1k2YRbF+ahJnJnQk2Pvrj92vbkSYUCCXvmqUvQRqZaPxb/cJX+R7hfiQS7GS/0rd51IzmFXPZUUirEKp2NKYlXfpxzqSQ6HY+KU29H06eX0uY+1TrgrOw8khwqx8uyTpbBSIkZpU3/3c2/lvCHJvN5/ppQ0fVqqUO0d3k85EpQd6gMljbojKRgxEFJAnDg8Tp3ioc9ICLJUv+t+OCUqn3r7c8XlQ+oJIYrIG9KheOWqiSRRl6YYMGax3VBITunU5QUhUmMjvUOUilrMJL1SWiRgezcht6OuffprJ4b0RbFK9Zw0xI5tTjn+GMWcyfHCge16lmTrzRadKh//p+T4HUJCTiM6LYegCvU4wYe01JOcxVNuivwT9ggfit2SSXFEAI9POWHIyYkbzVtsfeYDOe+pEcmhUElSR4R3Web/CZA8OJ90ofLMnn52QVJHGuHSyK78LocMjEU5F0uVh7qir5J4qW2aQwDmBmwomN6nGcm56anXM4mlZG6WDgywRd/LEzpdftZJMi4a+3kxESxFC8ucob6SHMRx/0SKGoU7+zKvIienL2h9c9cGyQmS09TaekNyqAD7XvxYPg69TVJxHH3uTdvktHuzSjPLpOMUq6TQCNIxWzrKPfDVQVlRkPMwWXJO8kXK1k6Sw4nvudxM+Wh7qWq8knJ9OCP1NeRxHLfB3OnE0ICcDNyo2Zm6/ribnsn5QRfvkJKL047Th06FNDizepLTEzaUlBvSYq7PZPR2EZjejIVrKAIcGJKyYPpumQrNxqVswocIzp/veymHO5wMOzWRnNLp1P4epGnkIVWH/Ts1gDOU51MfYqx/P4Tm/hfeSRg+m3fqMyeSIykVWaKarJ7IgVNikmm3TaR42XQCrT8NoTWGSiKslpK5hd3WPP7Orz5JUj67nuToU0k2SwJ7+xAlp57keLZk5ZdSuHDZ5KgnGW+0nDdDuWIn9STHKScn4xBuuSXX7IJcjZntygkyxO+qnRdP73lHLvSoPEK9kiOclnN0xhgl23DZjDMFymdcEA4FNcv3b4ZjI5LT7J7y53ukHCEqUjOSc2MiOXLHqLjUMMofUoXkDM5VGqzOtkpy9NVGY/603iA5iHwzm2zWxyA5zRAaPj8PkhMkp6nl9YbkSEa0cyfjU0Ts5nrTKBeUECcq1px78rzQOOEh1HNOWtxHTbt86sDhVz/5VaJkO0mOENV+K83ytVclux+WKjUrtEdRkTMxNDn/mhSy4jCmSQrMbmnHjJycm/IF5MYIC/nK+xRpt1+vWNT/YurIp0nip258e5QUTuoNiG28Rq7NDmc/mI/IOzWDBCoMV9sQIP068ZbnhpCcBRLJ6d149+VVJVNTGuRmOZb+SlLO5P04gbdFSjQ+IoWzZk2KgXouwhf9aUiVMWR3Tqnlz4UksifUVTZjOkOybWGf2iT2+t/Lns9N+Mg5UkZh+dkmzeoWh+yI+wfpu22TpgKGZUNCkByniST5OmmGJC+U+qlx9iXJmXbCMXNI6NyU1D1fUh+PSaqh/BqlB4TsnIrqTUM2YSvXbcmkKq2UTg02s83a9932rAcy2dh12RlyOLO3ra8kR5gKyTHGjVpfSI4ioUclrN796NOcyyWUvX4KJfqERa1K29u+1F8XJKdV5IbtfUFyguQ0tbDekJymD+nhArtgsX1SOcfeqElaPfTKgTlhUi7JN0lykJvrd188f1h0aI3T8IcjHE2uS7qY46FAccyHpiO6fvZNk5a+jot8HfVOTrv9+ayecJr1OT+1JIcCd8omlJz2kxw7d8eYD0ljL78DiRYO4cz3TTkfV6XE8tkTIZCX05Pt9LX/5fVscp0T7yqeTInRZdE/xGbvdNSYCoHM99TeTSqJWkVXPvpaDnUiykMjyE4JClepdTRnInNHrT1ntiFzQuhrl/Mf/orkTJ+OiMsDeuKVQfk95kqVm+U2wQoGCHVvm42J+6iYvVVxPJuadkNSWhA1IdjebmjcOzxJjnpQ8nIkQm+bPvcgAd37I6DtaEFy2oFi+58RJCdITlOrGpYkp+kvTxc4VePIq/yd2jyA3tzb0zVlTk4zJae3JKfR76k9at0XJ9KffrXjXrtjJ7A4cjvp+h2+pGNKzkmUnGFIcvQlJ5Hn/KXPi2Ovfybns+yQcnCEQ70n1W9YEEe/k7rCKQozaVSjK1ItnN4oHt4NITfuzd5Pkri8m+OSkiPXaKeU0yafxTvI69q1huSozuvZbKv+xGA7xr63z8jYNzjZ2Oz+4UlyhFdPT+Ppv/L+fA8OAW1XC5LTLiTb+5wgOUFymlrU8CY5XrA/eQCNOojkqBjsuKnQR30rw1X9ITlNge3gCyg6tceJa1/1jfeGkJwUNqHknDyMlJza38mpUpCE85yO6ik/ql2QIldqs7z94Se5fILPRVAuyhNK7fo9noPMIJUDU0G82VM+mhNag4tbfpnDSQpHCgk5Jt/fgnXtfO9WntVJJEdYUlmBdrUgOe1Csr3PCZITJKepRakWK98kfz5gw3maXj8sLmg3ybkuhRMUK1OLRHy+vin0JiwgRLFlOpIqLyjaYASQHMetT/4GSc7wxB4BURhQuFHtoGHRKJTCTkIntUpRWdcI2XZkvEqKYCOckByhLlW6hXAbNcUAy8Tjdubk1Co52w9RcoLkDAtr7qxnBskJktPUItU2UU/DDtrJk+HR2k1yOBVJzpxGo/AD1cAfp1I4mN6EKIYHLsPjd6oRpBjciEJyhgfG3fo71fR54Z0Pc6L4UjNNNFSSY3OF2LUr8dg6dkY62q8KszIXTmz5Un27Wig57UKyvc8JkhMkp1cWNfhj9V+vLtyrm9twUbtJThteaYR9hBo6cnIcsXcC65h0csyOO1og0AwBxR+FHn0KREiuUZPAr9ilOkg+OeGI95I9EKK+nK7yeREKLpIjF2/eqcZvW9KxfgTJaTb6w+fnQXKC5Awfy+vjbw2S00fAhuHliq5d+vDL+fMLjpdvnvKahnbaaBi+Sjy6CxGQoyRf7oD0dXgJ5j6K29OR/XqSQ3EVXt6/QVmIMq/ryVffz2UI2n0iL0hOZxpjkJwgOZ1pmXVvFSSns4ZJrsg7KSnXh09HGw71fDoLjXibYYFA+QX4oRXpozD74KmTl8i3nKYtEulWq6qnNliTbn8LktN+TNvxxCA5QXLaYUfD/BlBcoY5xH36BXbb/gg59LdSbJ9+cVwcCNQh4CSg78DlCtspx0717fKTEt8kWEFyvkm0e/+7guQEyem9tQzHK3MF5JQH8vqgT4o10rem+lvhdjh2JX51IBAItBkBig6FRmulfk87XidITjtQbP8zguQEyWm/VQ2DJ9qlOe00KFWTnSTVK/HhvmiBQCAQCHQKAkFyOmUk/vc9guQEyelMy4y3CgQCgUCgQggEyenMwQqSEySnMy0z3ioQCAQCgQohECSnMwcrSE6QnM60zHirQCAQCAQqhECQnM4crCA5QXI60zLjrQKBQCAQqBACQXI6c7CC5ATJ6UzLjLcKBAKBQKBCCATJ6czBCpITJKczLTPeKhAIBAKBCiEQJKczBytITpCczrTMeKtAIBAIBCqEQJCczhysIDlBcjrTMuOtAoFAIBCoEAJBcjpzsILkBMnpTMuMtwoEAoFAoEIIBMnpzMEKkhMkpzMtM94qEAgEAoEKIRAkpzMHK0jOkHH55S9/WVx//fXF/PPP36eR+vzzz4v//Oc/xVhjjVWMNNJIfbp3RLr4gw8+KEYbbbRi1FHjcwxDG/dPP/20+O9//1uMOeaYI5J59KmvMed6D9dHH32U1yVzL1pjBNo155599tli9NFHL84///yAuoMQCJIzZDBuvfXW4vHHHy/GGGOMPg3Pa6+9lo168803L8Ybb7w+3TsiXfynP/2pWHDBBYtZZpllROp2n/v6wAMPFP/4xz+Kddddt8/3jig3vPrqq4Vd82abbRZzrsmgX3HFFcWAAQOKH/zgByOKefS5n+bcc889V6yzzjp9vrf2BmTpu9/9brH22mv36zlxc3sRCJLTTzwHDhxYbLPNNnnRnWyyyfr5tO69ff31189Oafnll+/eTrahZwgzwn388ce34Wnd+Ygnnnii2G677fKcm3TSSbuzk23q1X777VdMOeWUGa9ojRE477zzittvv734wx/+EBB1IQJBcvo5qEFyegdgkJze4RQkpzlOQXKaY1ReESSnOVZBcppjVOUrguT0c/SQnO23374wUULJ6RnMDTfcsNh0002L5ZZbrp+Id/ftF1xwQd5VHnfccd3d0X70zpzbYYcd8pwLJWfoQB5wwAHFFFNMUWy77bb9QLy7b7WxuOOOO2LOdekwB8np58DKyeGYIj9g6ECeccYZOSdn5pln7ifi3X37gw8+mHNy+psf0M0olXNOHty4447bzV3td9/k5HznO98pFl100X4/q1sfICfnn//8Z+TSdOkAB8np58BKNnv99dfzjnKUUUbp59O693bJohySU2jRekbg/fffL5yImWiiiQKmHhCIOdd703jrrbfyicY4FBFzrvdW011XBsnprvGM3gQCgUAgEAgEAoHAEASC5IQpBAKBQCAQCAQCgUBXIhAkp8Vh/fjjj4t33nmnePvtt/MTxL0nnHDCri+69dlnnxUvvfRSIazyrW99q5h44olzaOXLL78sFGn717/+lYsj+hmJfJJJJilGHnnkjBHpHGaffPJJ8e1vf7uYfPLJc10ixcqEaGD573//O18LS5hWtYgZnN58883i3XffLb744ovcx/HHHz/X0YCHfvq562Dxve99L4fy4Ma25J0ooKhNMMEE+V7XeZZnulfYxj2eWcVwBJvRF6FMf/dn7LHHzv1lFx9++GG2NXbFDtiEmi8w8m9+5hr36T9bLItNsjN/2BXchJM9uxsKdr7xxht5rigYOdVUU2U7YC/vvfde7p9+mlv+Dhu2Zu6Zd2xPInJZvJSt+Vk572Bv3ilqV8XGJvSlHGe2MvXUU+f+9GbOletat865Ko5pf985SE6LCDrGeuGFFxYXX3xxnlArrLBCLgg4wwwz5EW4WxuH9Itf/KK47bbbsqPZaaedip/85Ce5Sq+Fd//99y8ee+yxnJ+01FJLFbvttlt2wq6VfPzXv/61ePHFF/NJNNfOPffcebF+6KGHct2TK6+8MkPnyLnk2+mnn76SUHJEJ510UnHddddlR6yPK620UrHxxhsX44wzTnHppZcWp59+enbynPeBBx5YzDvvvNmZP/XUU8Wxxx5b3H///dmWYKHAmMUagbzooouKc845p3jllVeKOeaYI//cs6tmd2xGYuzRRx+dyZ7/n2+++Qon8WafffbirrvuKlQiR/Y45i222KJYZZVVsm3B7Wc/+1m2G/f+8Ic/LHbcccdcEwYRZEtw+vvf/54dvp+xx6pXktbXk08+OY//nHPOmWu7vPDCC8Vvf/vbfCoPoVtggQUyboii6y+55JLizDPPzPaCAB188MG5sru/szWFOq+55po8z8w5tlbVAwJ77713cdlll+U5htCxlRNOOCHbhbWnds79/Oc/L+aZZ54855588smMYaM5Z/5a58s5B3dzbsUVV6zcnKvkYtrPlw6S0wKAFlFHfJ955plMbEyoPffcs9hoo42KZZZZJu+EurVxOAgOh3r22WcXCy20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)
