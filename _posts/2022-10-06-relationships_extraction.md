---
layout: single
title: "Relationships extraction with BLOOM"
date: 2022-10-06
use_math: true
comments: true
[]: #classes: wide
toc: true
toc_label: "Table of Contents"
toc_icon: "scroll"
author_profile: false
excerpt: This post briefly demonstrates how a large language model, when properly prompted, can be used to extract multiple types of relationships between entities.
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yabramuvdi/yabramuvdi.github.io/blob/master/_notebooks/relationships_extraction.ipynb)

*Can large language models extract relationships between entities in text data?*

This post briefly demonstrates how a large language model, when properly prompted, can be used to extract multiple types of relationships between entities.

------------


## Setup


```python
! pip install huggingface_hub
! git config --global credential.helper store
```

```python
from huggingface_hub import notebook_login
from huggingface_hub import HfFolder
from huggingface_hub import InferenceApi
import random
import time

from google.colab import output
output.enable_custom_widget_manager()
```


```python
# now we need to login to HuggingFace through a personal token.
# once you have a HuggingFace account (it's free) go to "settings" and then
# to "access tokens". There you can create a token for your notebooks
notebook_login()
```

    Login successful
    Your token has been saved to /root/.huggingface/token


## Text generation pipeline


```python
# initialize an inference object
# you could use provide your login token directly as a parameter
inference = InferenceApi("bigscience/bloom", token=HfFolder.get_token())
```


```python
# set up a pipeline for generating text from a prompt
def infer(prompt,
          max_length = 50,
          top_k = 0,
          num_beams = 0,
          no_repeat_ngram_size = 2,
          top_p = 0.9,
          seed=42,
          temperature=0.7,
          greedy_decoding = False,
          return_full_text = False):
    

    top_k = None if top_k == 0 else top_k
    do_sample = False if num_beams > 0 else not greedy_decoding
    num_beams = None if (greedy_decoding or num_beams == 0) else num_beams
    no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
    top_p = None if num_beams else top_p
    early_stopping = None if num_beams is None else num_beams > 0

    params = {
        "max_new_tokens": max_length,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "do_sample": do_sample,
        "seed": seed,
        "early_stopping":early_stopping,
        "no_repeat_ngram_size":no_repeat_ngram_size,
        "num_beams":num_beams,
        "return_full_text":return_full_text
    }
    
    s = time.time()
    response = inference(prompt, params=params)
    #print(response)
    proc_time = time.time()-s
    print(f"Processing time was {proc_time} seconds\n\n")
    return response
```


```python
# let's try a first simple example
prompt = "The thing that makes large language models interesting is"
resp = infer(prompt, max_length=50)
print(resp[0]["generated_text"])
```

    Processing time was 0.0995793342590332 seconds
    
    
    The thing that makes large language models interesting is that they are huge. I think the largest model we’ve seen was something like 1.2 billion parameters, which is much larger than the typical model that you’d train on a single GPU.
    It’s a bit of a chicken-and-egg problem.


## Relationship extraction

Now we can try to see if we can prompt the model to extract relationships between the entities present in some text. To do this we will start the prompt stating exactly what we want the model to do and then we will include three examples. This is known as *few-shot learning* in the literature.

### Family relationships


```python
# Simple example of family relationships
prompt = """Extract the people and their relationship in this text: 

Text: "Homer Simpson is married to Marge Bubie." 
Relationship: Homer Simpson, Marge Bubie, married 

Text: "In 1992, Maria Perez and John Smith celebrated their nuptials." 
Relationship: Maria Perez, John Smith, married 

Text: "Paul Rodriguez is the son of Julia Baena." 
Relationship: Paul Rodriguez, Julia Baena, child-parent 

Text: "Sammy Castro is the uncle of Nicolas Gonzalez." 
Relationship: """

resp = infer(prompt, max_length=100)
print(resp[0]["generated_text"])
```

    Processing time was 0.09788250923156738 seconds
    
    
    Extract the people and their relationship in this text: 
    
    Text: "Homer Simpson is married to Marge Bubie." 
    Relationship: Homer Simpson, Marge Bubie, married 
    
    Text: "In 1992, Maria Perez and John Smith celebrated their nuptials." 
    Relationship: Maria Perez, John Smith, married 
    
    Text: "Paul Rodriguez is the son of Julia Baena." 
    Relationship: Paul Rodriguez, Julia Baena, child-parent 
    
    Text: "Sammy Castro is the uncle of Nicolas Gonzalez." 
    Relationship:  Sammy Castro, Nicolas Gonzalez, uncle-nephew 
    
    Text: "Alexa Gonzalez is the granddaughter of Javier Gonzalez." 
    Relationship: Alexa Gonzalez, Javier Gonzalez, grandparent-grandchild 
    
    Text: "Sara Gonzalez is the aunt of Alejandro Gonzalez." 
    Relationship: Sara Gonzalez, Alejandro Gonzalez, aunt-nephew 
    
    Text: "Lucy Gonzalez is the daughter of Susan Gonzalez." 
    Relationship: Lucy Gonzalez, Susan Gonzalez, parent-child 
    
    Text: "Lucy Gonzalez is


It seems to work very well! And in addition to identifying the relationship in the example we provided, the model also generated some examples by itself.


```python
# multiple relationships
prompt = """Extract the people and their relationships in this text: 

Text: "Pedro Gonzalez is married to Ana De los Rios." 
Relationships: (Pedro Gonzalez, Ana De los Rios, married) 

Text: "Pablo Doe is the son of Carmen Rueda and Antonio Doe."
Relationships: (Pablo Doe, Carmen Rueda, child-parent), (Pablo Doe, Antonio Doe, child-parent)

Text: "Sara Cuadrado was born in 1995 from Daniela Muller and Juan Cuadrado."
Relationships: (Sara Cuadrado, Daniela Muller, child-parent), (Sara Cuadrado, Juan Cuadrado, child-parent) 

Text: "Tom Smith is married to Ana Dupont and son of the famous writer Charles Smith."
Relationships:"""

resp = infer(prompt, max_length=50)
print(resp[0]["generated_text"])
```

    Processing time was 3.722917079925537 seconds
    
    
    Extract the people and their relationships in this text: 
    
    Text: "Pedro Gonzalez is married to Ana De los Rios." 
    Relationships: (Pedro Gonzalez, Ana De los Rios, married) 
    
    Text: "Pablo Doe is the son of Carmen Rueda and Antonio Doe."
    Relationships: (Pablo Doe, Carmen Rueda, child-parent), (Pablo Doe, Antonio Doe, child-parent)
    
    Text: "Sara Cuadrado was born in 1995 from Daniela Muller and Juan Cuadrado."
    Relationships: (Sara Cuadrado, Daniela Muller, child-parent), (Sara Cuadrado, Juan Cuadrado, child-parent) 
    
    Text: "Tom Smith is married to Ana Dupont and son of the famous writer Charles Smith."
    Relationships: (Tom Smith, Ana Dupont, married), (Tom Smith, Charles Smith, child-parent)
    
    Text: "Maria Alvarez is the sister of Pedro Alvarez."
    Relationships: (Maria Alvarez, Pedro Alvarez, sister)
    
    Text: "Juan Perez is the


We can see that the model understood the new format in which we wanted to extract the relationships and was able to correct identify them. Now we want to further test the capacity of the model by combining text in English and Spanish.


```python
# multiple relationships with multilingual text
prompt = """Extract the people and their relationships in this text: 

Text: "Pedro Gonzalez is married to Ana De los Rios." 
Relationships: (Pedro Gonzalez, Ana De los Rios, married) 

Text: "Pablo Doe is the son of Carmen Rueda and Antonio Doe."
Relationships: (Pablo Doe, Carmen Rueda, child-parent), (Pablo Doe, Antonio Doe, child-parent)

Text: "Sara Cuadrado was born in 1995 from Daniela Muller and Juan Cuadrado."
Relationships: (Sara Cuadrado, Daniela Muller, child-parent), (Sara Cuadrado, Juan Cuadrado, child-parent) 

Text: "Doña Brigida Gómez de Orozco y Dominguez, esposa en primeras nupcias del Capitán Dionisio de Velasco, alférez mayor y encomendero de San Cristobal" 
Relationships: (Doña Brigida Gómez de Orozco y Dominguez, Capitán Dionisio de Velasco, married) 

Text: "Guillermo Acosta Acosta, casado con Ofelia Uribe Duran, de importantes logros políticos, periodista e institutora, hermana de Beatríz Uribe Duran." 
Relationships: (Guillermo Acosta Acosta, Ofelia Uribe Duran, married), (Ofelia Uribe Duran, Beatriz Uribe Duran, siblings)

Text: "Don Rafael Vergara, es el padre de Enrique Vergara, el presidente del Colegio de Abogados de Lima, y el esposo de Enriqueta María de los Milagros."
Relationships:"""

resp = infer(prompt, max_length=100)
print(resp[0]["generated_text"])
```

    Processing time was 8.13334321975708 seconds
    
    
    Extract the people and their relationships in this text: 
    
    Text: "Pedro Gonzalez is married to Ana De los Rios." 
    Relationships: (Pedro Gonzalez, Ana De los Rios, married) 
    
    Text: "Pablo Doe is the son of Carmen Rueda and Antonio Doe."
    Relationships: (Pablo Doe, Carmen Rueda, child-parent), (Pablo Doe, Antonio Doe, child-parent)
    
    Text: "Sara Cuadrado was born in 1995 from Daniela Muller and Juan Cuadrado."
    Relationships: (Sara Cuadrado, Daniela Muller, child-parent), (Sara Cuadrado, Juan Cuadrado, child-parent) 
    
    Text: "Doña Brigida Gómez de Orozco y Dominguez, esposa en primeras nupcias del Capitán Dionisio de Velasco, alférez mayor y encomendero de San Cristobal" 
    Relationships: (Doña Brigida Gómez de Orozco y Dominguez, Capitán Dionisio de Velasco, married) 
    
    Text: "Guillermo Acosta Acosta, casado con Ofelia Uribe Duran, de importantes logros políticos, periodista e institutora, hermana de Beatríz Uribe Duran." 
    Relationships: (Guillermo Acosta Acosta, Ofelia Uribe Duran, married), (Ofelia Uribe Duran, Beatriz Uribe Duran, siblings)
    
    Text: "Don Rafael Vergara, es el padre de Enrique Vergara, el presidente del Colegio de Abogados de Lima, y el esposo de Enriqueta María de los Milagros."
    Relationships: (Don Rafael Vergara, Enrique Vergara, father-son), (Don Rafael Vergara, Enriqueta María de los Milagros, married)
    
    Text: "María Fernanda de la Puente del Río, es la hija de la periodista y política, Elena de la Puente del Río y del ex Ministro de Educación, Alfonso de la Puente del Río."
    Relationships: (María Fernanda de la Puente del Río, Elena de la Puente del Río, child-parent), (María Fernanda de la Puente del


This is awesome! Notice that the model is able to generate new examples in Spanish.

### Occupations

Lastly, we want to see if the model is capable of identifying other types of relationships in the text. We will focus on extracting the occupation of people mentioned in the text.


```python
# simple occupation extraction
prompt = """Extract the people and their occupations in this text: 

Text: "Pedro Gomez is the son of Ofelia Uribe and the director of the largest hospital in Lima." 
Occupations: (Pedro Gomez, doctor)

Text: "Rafael Smith was known as the best butcher in the city."
Occupations: (Rafael Smith, butcher)

Text: "All his life, Nicolas Duran worked as a highschool teacher in multiple schools."
Occupations: (Nicolas Duran, teacher)

Text: "John Vega dedicated all his days to making the best bread in the city."
Occupations:"""

resp = infer(prompt, max_length=50, seed=random.randint(0, 100000))
print(resp[0]["generated_text"])
```

    Processing time was 3.8372151851654053 seconds
    
    
    Extract the people and their occupations in this text: 
    
    Text: "Pedro Gomez is the son of Ofelia Uribe and the director of the largest hospital in Lima." 
    Occupations: (Pedro Gomez, doctor)
    
    Text: "Rafael Smith was known as the best butcher in the city."
    Occupations: (Rafael Smith, butcher)
    
    Text: "All his life, Nicolas Duran worked as a highschool teacher in multiple schools."
    Occupations: (Nicolas Duran, teacher)
    
    Text: "John Vega dedicated all his days to making the best bread in the city."
    Occupations: (John Vega, baker)
    
    Text: "Jose Alvarez is the son of Catalina Gomez and a police officer."
    Occupations: (Jose Alvarez, police officer)
    
    Text: "Luciana Perez is the daughter of Matilde Lopez and a psychologist

