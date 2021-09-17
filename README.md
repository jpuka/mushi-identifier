# Mushi identifier

## Introduction

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

Can you eat either of them?

<br>

Have you ever wandered around in the beautiful autumn forest looking for fungi food, but ended up spending most of your time staring at a book while getting inhabited by moose flies. If so, this tool might be just for you! It will help you deduce which mushroom is delicious and which kills you, while allowing you to spend more time marvelling the nature around you.

Mushi-identifier is a smartphone app, that allows you to take a photo of a mushroom. The app recognizes the mushroom species in the image and returns the name of the mushroom with a confidence score.

The app should be used together with a recent mushroom book. If you find a mushroom you do not know, take a photo for the app and it will tell you what it looks like. Then, you can quickly find the mushroom in the book glossary instead of scrolling through endless pages looking for images of it.

The app is targeted at novice mushroom hunters, and for now it only identifies the edible mushrooms recommended by the [Finnish food authority](https://www.ruokavirasto.fi/henkiloasiakkaat/tietoa-elintarvikkeista/elintarvikeryhmat/ruokasienet/suositeltavat-ruokasienet/). These are common and easy to verify with a book even for beginners.

*NOTE*: This project is a work-in-progress. Check the TODO-section below for a quick overview of the development stage.

## Motivation

This is an autumn deep learning project, that I felt inspired to start after going mushroom hunting with friends with no prior experience in mushrooms. I found that especially for people with less experience, the majority of the time in the forest is spent staring at a book trying to find images that resemble the mushroom in front of you.

During my initial survey I found plenty of deep learning projects that attempt to classify mushrooms as edible vs. non-edible. For me this is nonsensical.

For me it's nonsensical to build an app that simply tells if a mushi is edible or not. This does not support learning, makes every wrong prediction potentially fatal. However, I have many times been in a situation where I spend most of my mushroom hunting time scrolling through the pages of a book looking for any look-likes. This is the problem this project seeks to alleviate, by doing it for you. Then you can check in the glossary and find it quicker.

This can make to help you identify mushrooms, especially if you really have no idea what you might be looking for. So it makes it quicker to find an unknown species in a book.

What makes this special is that it focuses on common species in Finland. Furthermore, the base training dataset is fresh and robust.

## Technical details

### Data

We are using the Danish Fungi 2020. Very neat, but unbalanced.

We started the project with another Danish dataset and were planning to complement it with scraped data. However, now this is set as external data and used to add images to missing classes.

### Model

Mushi-identifier is built on a convolutional neural network. The image recognition task is defined as single-label multi-class classification, since the user is expected to submit only one mushroom species in each image.

Due to a shortage of data, I am using transfer learning with fine-tuning. The base CNN is mobilenet, taught with ImageNet. MobileNet is light enough to run on mobile devices, which are target deployment surface. ImageNet already has elementary mushroom knowledge, which helps with the task.

### Deployment

The deployment is done as a mobile app, since mushroom places tend to be low connectivity environments. However, a REST api + Flask version will also be developed and deployed on a web server as a practice exercise.

The packaging / dependency manager is [Poetry](https://python-poetry.org/), since it is modern and practical and follows the build system standard set by [PEP-517](https://www.python.org/dev/peps/pep-0517/).

## Folder structure

```bash
├── data
│   ├── 00_external        # Web-scraped images, mushroom classes
│   ├── 00_raw             # Danish dataset: images and metadata
│   ├── 01_interim         # Clean, non-corrupted data from external & raw
│   └── 02_processed       # Structured model-ready data
├── docs
│   └── images             # Images for this README
├── models                 # Saved models
├── notebooks              # Jupyter notebooks (EDA, model presentation)
└── src
    ├── data               # Python code for data manipulation (scraping, cleaning, loading)
    └── model              # Python code for model training and predictions
```

## Disclaimer

Please don't fully trust any image recognition software for classifying mushrooms. Models such as the one in mushi-identifier can help you, but they cannot replace an experienced friend and a recent mushroom book.

Mushrooms often have features you can only learn by feeling, peeling, cutting and smelling them. Things like this are difficult or impossible to teach to image recognition software. Also, habitat and time of year matters.

The software can aid you, if you carry a mushroom book with you and verify stuff.

## TODO

### README

* Polish text
* Make mushroom images and titles in GIMP: insert single image to markdown

### EDA

* Visualize a couple of images with labels to get an idea for the data quality

### Data

* Create a script that uses tf.keras.utils.get_file to get data
* Start with the classes with 100+ examples, add more classes later
* Add fig show in notebook
* Rethink if "interim" is needed
* Visualize validation data in notebook (imshow)
* If unbalanced, scrape for more data
  * Add the scraped data to external, with same ids and folder styles as raw
  * Scraped data should be of similar distribution as raw - non-professional photos
  * Make sure to do a balanced division for the external data into train, validation, test 
* Resplit train-validation (-test?) data, only 3 validation mushrooms per class now
* Scrape for more training data [long-term]
* Add more mushroom categories

### Model

* Search Arxiv for mushroom identifiers -> what model types did others use.
* Estimate a random baseline to compare model accuracy to
* Implement k-fold cross validation instead of standard data split to increase the reliability of validation scores.
  Consider doing iterated k-fold with shuffling, if enough computational resources. If we do hyperparameter tuning, this might be needed in any case to not overfit to the validation data.
* Based on the model prediction, present questions to the user ("if you cut it, does it bleed white? y/n") to verify the species

### Deployment

* Study cybersecurity best practices on how to deploy the model on a web server - allowing the user to upload images.
* Use Tensorflow lite to deploy model on smartphone, due to usage on low-connectivity environments
* Do weight pruning and quantization to optimize the model before deployment on smartphone.

Put this into issues?

- [x] Mushi 1
- [ ] Mushi 2
- [ ] Mushi 3