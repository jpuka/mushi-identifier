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

Mushi-identifier is an image-recognition application (web & mobile) that identifies mushroom species from photos submitted by the user. After receiving a photo, the app runs it through a convolutional neural network (CNN) and returns the names and confidence scores of the top three mushrooms that most closely resemble the species in the image.

Mushi-identifier's predictions should always be verified with a recent mushroom book. The core idea of the app is to allow the user to quickly find the spotted mushroom in the book's glossary - instead of browsing through endless pages looking for images of the species, the user can simply check the app's three predictions in the order of confidence. 

The app is targeted at novice mushroom hunters, and for now it seeks to identify the 26 edible mushrooms species recommended by the [Finnish food authority](https://www.ruokavirasto.fi/henkiloasiakkaat/tietoa-elintarvikkeista/elintarvikeryhmat/ruokasienet/suositeltavat-ruokasienet/). These species are common and easy to verify with a book even for beginners.

*Note: Mushi-identifier is a work-in-progress. Currently, a baseline model has been trained with the raw dataset (21/26 species) and deployed as a REST API with Docker and FastAPI, and will soon be made public on a server. Check the [Roadmap](#6-roadmap) below for an overview of the development stage.*

## 2 Motivation

In the autumn of 2021, I did a lot of mushroom hunting trips with friends who were totally new to the sport. While many fungi were found, I realized that due to our combined curiosity we spend most of our time flipping through the pages of various mushroom books. The trips were still very fun and great for learning, but because I prefer to stare at colorful trees in the nature, and have a boring physicist's brain, I wanted to find a way to make the mushi identifying more efficient.

By this time I had already been studying neural networks for over a year, so I figured I could solve the problem in a computer vision (CV) project. I started with a review of existing CV identifier apps. Some of the apps I found identified mushrooms as edible/non-edible which I found both detrimental to learning and dangerous - identifying mushrooms involves feeling, peeling, cutting and smelling. Others looked very promising with inbuilt descriptions but they were closed-source with ads and in-app purchases. This is when I started feeling the urge to develop an open-source app that would be truly free for any enthusiastic mushroom pickers.

I wanted to build the app from the core to teach people to use mushroom books. I would design it to help beginners get into this great hobby, and its goal would be to eventually become obsolete to its users - once they learned to fully rely on a book and their experience. I decided to focus and tune the application on species common in Finnish conditions and have it return multiple suggestions for each photo to improve its utility. Overall, I felt excited to start a project that could be both a fun learning exercise and useful in practice.

And so, mushi-identifier was born.

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
# Start (the --rm flag is optional, but helpful for testing)
docker run -d -p 8000:8000 --name mia --rm mushi-identifier-app
# Check IP if needed (IPAddress)
docker inspect mia
# Connect to ip:8000 on browser
```

## 4 Project structure

The project structure is loosely based on the Cookiecutter data science [link] template.

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

I am using the [Danish Fungi 2020 dataset](https://arxiv.org/abs/2103.10107) (preprint paper). Very neat, but unbalanced / long-tailed. Good, more realistic dataset, since uniformly distributed data is a rarity anyway. Read all about it there, but to sum it up: sumuppp.

The raw dataset contains images for 21 out of 26 classes. See EDA for distribution. The data for the remaining classes will be scraped from sources such as. Furthermore, classes with a low image count might be completed with scraped images.

I started the project with another Danish dataset and were planning to complement it with scraped data. However, now this is set as external data and used to add images to missing classes. [iNaturalist](https://github.com/visipedia/inat_comp/tree/master/2017#Data) dataset.

### Model

Mushi-identifier is built on a convolutional neural network. The image recognition task is defined as single-label multi-class classification, since the user is expected to submit only one mushroom species in each image.

Due to a shortage of data, I am using transfer learning with feature extraction. I will eventually do fine-tuning to improve the performance. The base CNN is mobilenet, taught with ImageNet. MobileNet is light enough to run on mobile devices, which are target deployment surface.

I chose MobilenetV2 since it is fast to train - I currently don't have supercomputers at my disposal I am working with Google Colab GPUs. Furthermore, it is light enough to be deployed mobile devices.

### Deployment

The deployment is done as a mobile app, since mushroom places tend to be low connectivity environments. However, a REST api + Flask version will also be developed and deployed on a web server as a practice exercise.

The packaging / dependency manager is [Poetry](https://python-poetry.org/), since it is modern and practical and follows the build system standard set by [PEP-517](https://www.python.org/dev/peps/pep-0517/).


## 6 Roadmap

This roadmap provides a quick overview of the project development stage. The roadmap will be updated as the project progresses.

_About this: For a larger project with multiple developers I would use a proper project management environment that links the roadmap to issues/commits. For this project, I find that having the roadmap here is sufficient and easiest for the readers._

### Data

**Base steps**
- [x] Review literature and find a solid raw (base) dataset
- [x] Do EDA on the raw dataset
- [x] Verify non-corruption and transfer *raw* data to *interim*
- [x] Split (train/validation/test) and transfer *interim* data to *processed*
- [x] Import *processed* data to tensorflow and start developing the model

**Additional steps**

- [ ] Write a get-script for loading the raw dataset with a MD5/SHA sum check 
- [ ] Scrape *external* data from [iNaturalist](https://www.inaturalist.org/), [Danmarks Svampeatlas¹](https://svampe.databasen.org/), [Luontoportti](https://luontoportti.com/) and/or [GBIF](https://www.gbif.org/).
  - [ ] Scrape data for mushroom species missing from the raw dataset (Albatrellus ovinus, Hygrophorus camarophyllus, Morchella spp., Russula vinosa, Tricholoma matsutake)
  - [ ] Scrape additional data for species with a low image count in the raw dataset
- [ ] Do EDA on the scraped external datasets
- [ ] Verify and add *external* data to *interim* mixing it with the raw data²
- [ ] Split (train/validation/test) and transfer mixed *interim* data to *processed*
- TODO: Combine above two steps
- [ ] Import supplemented *processed* data to tensorflow and use it to improve the model
- [ ] Investigate the extra images present in the raw dataset

¹ The raw dataset is from Svampeatlas, so avoid scraping duplicate images, that could get split to both train and test sets biasing the test set.  
² Think about this - the validation/test sets will need to be of the same distribution.

### Model

**Base steps**

- [x] Review literature and make initial modelling choices (architecture, metrics, baseline performance, hyperparameters)
- [x] Build, train and save a baseline model
- [x] Write functions for plotting and prediction
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
- [ ] Mobile app: Wrap the model in an app and deploy on mobile

**Additional steps**

- [ ] Web app: Implement a simple frontend for the web API, so it is easy to use with a browser.
- [ ] Based on the model prediction, present yes/no-questions to the user ("If you cut the bottom, does it bleed white? y/n") to help verify the species.