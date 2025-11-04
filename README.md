# 7th Place Solution for Spanish/French language classification contest

### Goal: 
Build a model that classifies a word as either Spanish or French. 

### Background: 
This competition is part of the course curriculum at UCSD for the class DSC 140A: Probabilistic Modeling and Machine Learning. No machine learning libraries were allowed for this competition. All machine learning models had to be built from scratch. 100+ students took part in this competition. A portion of our final class grade depended on the ranking in this competition. 

### Outcome: 
2nd out of 109 student; RMSE - 82.40811. First place RMSE - 82.32490.

### Approach: 
Linear soft-margin Support Vector Machine with feature selection based on minimal entropy. Several models were developed and tested, the submitted model can be found in classify.py.
