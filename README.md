# Mushi identifier

## Introduction

Which mushroom is this?

<img src="docs/images/example_russula_claroflava.JPG" alt="russula claroflava - keltahapero" width="200"/>

How about this?

<img src="docs/images/example_lactarius_torminosus.jpg" alt="lactarius torminosus - karvarousku" width="200"/>

Can you eat either of them?

If you have ever wandered around in the autumn forest surrounded by fungi food, but unable to deduce which mushroom is delicious and which kills you, this tool might be just for you!

Mushi-identifier accepts user submitted images and recognizes the mushroom species in them. It returns the name of the mushroom with the recognition probability.

## Motivation

This is an autumn deep learning project, that I felt inspired to start after going mushroom hunting with friends with no prior experience in mushrooms.

I had been studying deep learning throughout my gap year and wanted to make a practical project, but was lacking ideas. Once I came up with this idea, I immediately felt inspired.

## Technical details

Mushi-identifier is built on a convolutional neural network. The image recognition task is defined as single-label multi-class classification, since the user is expected to submit only one mushroom species in each image.

The dataset is a mixture between the photos in the Danish mushroom dataset and images scraped from the internet. The Danish set is a good starting point, but unbalanced and lacks validation data (see EDA).

Due to a shortage of data, I am using transfer learning with fine-tuning. The base CNN is mobilenet, taught with ImageNet. MobileNet is light enough to run on mobile devices, which are target deployment surface. ImageNet already has elementary mushroom knowledge, which helps with the task.

The deployment is done as a mobile app, since mushroom places tend to be low connectivity environments. However, a REST api + Flask version will also be developed and deployed on a web server as a practice exercise.

The packaging / dependency manager is [Poetry](https://python-poetry.org/), since it is modern and practical and follows the build system standard set by [PEP-517](https://www.python.org/dev/peps/pep-0517/).

## Disclaimer

Please don't fully trust any image recognition software for classifying mushrooms. Models such as the one in mushi-identifier can help you, but they cannot replace an experienced friend and a recent mushroom book.

Mushrooms often have features you can only learn by turning, peeling, cutting and smelling them. Things like this are difficult or impossible to teach to image recognition software.

The software can aid you, if you carry a mushroom book with you and verify stuff.

## Folder structure

## TODO

### EDA

* Visualize a couple of images with labels to get an idea for the data quality

### Data

* Start with the classes with 100+ examples, add more classes later
* Add fig show in notebook
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