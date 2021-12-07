This is my source code for my master at École de Technologie Supérieure in partnership with Desjardins.

The goal was to create a proof of concept to determine if it is possible to anonymize french audio recordings.

The code is separated as followed :
- fa : Code used to evaluate two forced alignment algorithms
- ner : Code used to train and evaluate three named entity recognition models
- pipeline : Code used to create the docker image to anonymize audio recordings