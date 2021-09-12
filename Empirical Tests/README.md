## Empirical tests

The empirical tests seek to compare selected tools concerning model performance, computation time, and usability. The selected tools, as mentioned in the [Wiki](https://github.com/m-rosso/autoML/wiki/AutoML---Discussion) of this project, are the following: [Auto-sklearn](https://automl.github.io/auto-sklearn/master/), [TPOT](https://epistasislab.github.io/tpot/), [MLJAR](https://mljar.com/) and [PyCaret](https://pycaret.org/).

In addition to the notebooks that more carefully present how to use each selected Python library, a notebook with analysis of results discusses the findings in terms of the criteria mentioned above. Finally, these empirical tests make use of a dataset found in Kaggle repository of datasets, which consists of a dataset for binary classification whose objective is to construct a classification algorithm for the [identification of malware apps](https://www.kaggle.com/saurabhshahane/android-permission-dataset).

If asked about the best tool among all the tested ones, MLJAR has been a great surprise, specially because of its different modes made specially for distinct types of users. Even so, Auto-sklearn and TPOT (specially the last one) deserve attention given their robust search space that eventually would perform better than MLJAR in a wider range of datasets.
