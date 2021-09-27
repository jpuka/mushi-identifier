# Mushi-identifier

## Contents

[1 Introduction](#1-introduction)  
[2 Motivation](#2-motivation)  
[3 Installation](#3-installation)  
[4 Project structure](#4-project-structure)  
[5 Technical details](#5-technical-details)  
[6 Roadmap](#6-roadmap)


## 1 Introduction

<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <p>Which mushroom is this?</p>
    <img alt="Forwarding" src="docs/images/example_russula_claroflava.jpg" height="250">
  </p> 
</td>
<td> 
  <p align="center">
    <p>How about this one?</p>
    <img alt="Routing" src="docs/images/example_lactarius_torminosus.jpg" height="250">
  </p> 
</td>
</tr></table>

Can you eat either of them? TODO: Make mushroom images and titles a single image in GIMP

<br>

Have you ever wandered around in the beautiful autumn forest looking for fungi food, but ended up spending most of your time staring at a book while getting inhabited by deer flies. If so, this tool might be just for you! It will help you deduce which mushroom is delicious and which kills you, while allowing you to spend more time marvelling the nature around you.

Mushi-identifier is an image-recognition application (web & mobile) that recognizes mushroom species from photos. After receiving a photo, it returns the names of the top-3 mushrooms it resembles with a confidence score.

The app should be used together with a recent mushroom book. If you find a mushroom you do not know, take a photo for the app and it will tell you what it looks like. Then, you can quickly find the mushroom in the book glossary instead of scrolling through endless pages looking for images of it.

The app is targeted at novice mushroom hunters, and for now it identifies the 26 edible mushrooms species recommended by the [Finnish food authority](https://www.ruokavirasto.fi/henkiloasiakkaat/tietoa-elintarvikkeista/elintarvikeryhmat/ruokasienet/suositeltavat-ruokasienet/). These species are common and easy to verify with a book even for beginners.

*NOTE: This project is a work-in-progress. Currently, a baseline model trained with the raw dataset (21/26 species) is ready and deployed as a REST API with Docker and FastAPI. Check the [Roadmap](#6-roadmap) below for an overview of the development stage.*

## 2 Motivation



This is an autumn deep learning project, that I felt inspired to start after going mushroom hunting with friends with no prior experience in mushrooms. I found that especially for people with less experience, the majority of the time in the forest is spent staring at a book trying to find images that resemble the mushroom in front of you.

During my initial survey I found plenty of deep learning projects that attempt to classify mushrooms as edible vs. non-edible. For me this is nonsensical.

For me it's nonsensical to build an app that simply tells if a mushi is edible or not. This does not support learning, makes every wrong prediction potentially fatal. However, I have many times been in a situation where I spend most of my mushroom hunting time scrolling through the pages of a book looking for any look-likes. This is the problem this project seeks to alleviate, by doing it for you. Then you can check in the glossary and find it quicker.

This can make to help you identify mushrooms, especially if you really have no idea what you might be looking for. So it makes it quicker to find an unknown species in a book.

What makes this special is that it focuses on common species in Finland. Furthermore, the base training dataset is fresh and robust.

*Write some existing apps here. This is a project focusing on Finnish mushrooms. This is a practice project.*

### A word of caution

Mushroom identification techniques include feeling, peeling, cutting and smelling the fungi. Furthermore, the habitat, nearby tree species and the time of the year also affect the identification. Features like these are difficult or impossible to teach to an image recognition software.

Therefore, please don't blindly trust any image recognition application for classifying mushrooms. Apps such as mushi-identifier can be helpful, but they cannot replace an experienced friend and/or a recent mushroom book. Even a well-trained model will sometimes make false predictions.

That being said, as long as you use the mushi-identifier together with some healthy scepticism and a good mushroom book, it should save you a lot of time and make your fungi trips fun and pleasant.

## 3 Installation

git pull repo  
install poetry


### Project

1. poetry install reqs
2. get data with "get_..." scripts, depending on what you want to load
3. cd to directory
4. start running scripts and stuff.

### App

1. install docker

```bash
# Folder
cd app
# Build
docker build -t mushi-identifier-app .
# Start (--rm flag optional, but can help while modifying)
docker run -d -p 8000:8000 --name mia --rm mushi-identifier-app
# Check IP if needed (IPAddress)
docker inspect mia
# Connect to ip:8000 on browser
```

## 4 Project structure

Based on Cookiecutter data science with some modifications.

```bash
├── data
│   ├── 00_external        # Web-scraped images, mushroom classes
│   ├── 00_raw             # Danish Fungi 2020 dataset: images and metadata
│   ├── 01_interim         # Non-corrupted species-wise data combined from external & raw
│   └── 02_processed       # Model-ready data split into train/validation/test from interim
├── docs
│   └── images             # Images for this README
├── models                 # Saved models
├── notebooks              # Jupyter notebooks (EDA, model presentation)
└── src
    ├── data               # Python code for data manipulation (scraping, cleaning, loading)
    └── model              # Python code for model training and predictions
```

## 5 Technical details

### Data

I am using the [Danish Fungi 2020 dataset](https://arxiv.org/abs/2103.10107) (preprint paper). Very neat, but unbalanced / long-tailed. Good, more realistic dataset, since uniformly distributed data is a rarity anyway.

The raw dataset contains images for 21 out of 26 classes. See EDA for distribution. The data for the remaining classes will be scraped from sources such as. Furthermore, classes with a low image count might be completed with scraped images.

I started the project with another Danish dataset and were planning to complement it with scraped data. However, now this is set as external data and used to add images to missing classes. [iNaturalist](https://github.com/visipedia/inat_comp/tree/master/2017#Data) dataset.

### Model

Mushi-identifier is built on a convolutional neural network. The image recognition task is defined as single-label multi-class classification, since the user is expected to submit only one mushroom species in each image.

Due to a shortage of data, I am using transfer learning with fine-tuning. The base CNN is mobilenet, taught with ImageNet. MobileNet is light enough to run on mobile devices, which are target deployment surface.

This [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Sulc_Fungi_Recognition_A_Practical_Use_Case_WACV_2020_paper.pdf).

### Deployment

The deployment is done as a mobile app, since mushroom places tend to be low connectivity environments. However, a REST api + Flask version will also be developed and deployed on a web server as a practice exercise.

The packaging / dependency manager is [Poetry](https://python-poetry.org/), since it is modern and practical and follows the build system standard set by [PEP-517](https://www.python.org/dev/peps/pep-0517/).


## 6 Roadmap

This simple roadmap provides a quick overview of the project development stage. The roadmap will be updated as the project progresses.

_About this: For a larger project with multiple developers I would use a proper project management environment that links the roadmap to issues/commits. For this project, I find that having the roadmap here is sufficient and easiest for the readers._

### Data

**Base steps**
- [x] Review literature and find a solid raw (base) dataset
- [x] Do EDA on the raw dataset
  - compare our classes to ones in article, numbers, counts
  - make guesses how our model will behave
- [x] Verify non-corruption and transfer *raw* data to *interim*
- [x] Split (train/validation/test) and transfer *interim* data to *processed*
- [x] Import *processed* data to tensorflow and start developing the model

**Additional steps**

- [ ] Create a get-script for loading raw data with tf.keras.utils.get_file (add a md5sum check)
- [ ] Scrape *external* data from [iNaturalist](https://www.inaturalist.org/), [Danmarks Svampeatlas¹](https://svampe.databasen.org/), [Luontoportti](https://luontoportti.com/) and/or [GBIF](https://www.gbif.org/).
  - [ ] Scrape data for mushroom species missing from the raw dataset 
    - Albatrellus ovinus, Hygrophorus camarophyllus, Morchella spp., Russula vinosa, Tricholoma matsutake
  - [ ] Scrape additional data for species with a low image count in the raw dataset
- [ ] Do EDA on the scraped external datasets
- [ ] Verify and transfer *external* data to *interim* mixing it with the raw data²
- [ ] Split (train/validation/test) and transfer mixed *interim* data to *processed*
- TODO: Combine above two steps
- [ ] Import supplemented *processed* data to tensorflow and use it to improve the model

¹ The raw dataset is from Svampeatlas, so avoid scraping duplicate images, that could get split to both train and test sets biasing the test set.  
² Think about this - the validation/test sets will need to be of the same distribution.

### Model

**Base steps**

- [x] Review literature and make initial modelling choices (architecture, metrics, baseline performance, hyperparameters)
- [x] Build, train and save a baseline model
- [x] Write prediction and plotting functions
- [ ] Implement k-fold cross-validation to increase reliability of validation metrics and to allow hyperparameter tuning without overfitting validation data
- [ ] Tune hyperparameters
- [ ] Build, train and save an improved model

**Additional steps**
 
- [ ] Add macro-averaged F1 score to metrics, since it works well for long-tailed class distributions
- [ ] Try MobileNetV3 as the base model
- [ ] Try fine-tuning instead of feature extraction, once there is enough data

### Deployment

**Base steps**

- [x] Web app: Implement a simple REST API with Docker and FastAPI
- [ ] Web app: Study security best practices and make the API public on a VPS.
- [ ] Mobile app: Optimize model for mobile - do weight pruning and quantization, convert to Tensorflow lite
- [ ] Mobile app: Deploy the model on mobile and develop the app

**Additional steps**

- [ ] Web app: Implement a simple frontend for the web API, so it is easy to use with a browser.
- [ ] Based on the model prediction, present yes/no-questions to the user ("If you cut the bottom, does it bleed white? y/n") to verify the species.