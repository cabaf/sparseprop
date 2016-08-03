# SparseProp: Temporal Proposals for Activity Detection.

This project hosts code for the framework introduced in the paper: **[Fast Temporal Activity Proposals for Efficient Detection of Human Actions in Untrimmed Videos](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Heilbron_Fast_Temporal_Activity_CVPR_2016_paper.pdf)**

The paper introduces a new method that produces temporal proposals in untrimmed videos. The method is not only able to retrieve temporal locations of actions with high recall but also it generates proposals quickly.

![Introduction Figure][image-intro]

If you find this code useful in your research, please cite:

```
@InProceedings{sparseprop,
author = {Caba Heilbron, Fabian and Niebles, Juan Carlos and Ghanem, Bernard},
title = {Fast Temporal Activity Proposals for Efficient Detection of Human Actions in Untrimmed Videos},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}
```

# What to know before starting to use SparseProp?
* **Dependencies:** SparseProp is implemented in [Python 2.7](https://www.python.org/download/releases/2.7/) including some third party packages: [NumPy](http://www.numpy.org/), [Scikit Learn](http://scikit-learn.org/), [H5py](http://www.h5py.org/), [Pandas](http://pandas.pydata.org/), [SPArse Modeling Software](http://spams-devel.gforge.inria.fr/), [Joblib](https://pythonhosted.org/joblib/).

* **Installation:** If you installed all the dependencies correctly, simply clone this repository to install SparseProp: ```git clone https://github.com/cabaf/sparseprop.git```

* **Feature Extraction:** The feature extraction module is not included in this code. The current version of SparseProp supports only [C3D](http://vlg.cs.dartmouth.edu/c3d/) as video representation.

# What SparseProp provides?
* **[Pre-trained model](https://raw.githubusercontent.com/cabaf/sparseprop/master/data/class_induced_thumos14.pkl):** Class-Induced model trained using videos from Thumos14 validation subset. Videos are represented using [C3D](http://vlg.cs.dartmouth.edu/c3d/) features.

* **[Pre-computed action proposals](https://drive.google.com/open?id=0B9WpeMTDrC3fdWJjajhuODZXS3c):** The resulting temporal action proposals in the Thumos14 test set.

* **Code for retrieving proposals in new videos:** Use the script ```retrieve_proposals.py``` to retrieve temporal segments in new videos. You will need to extract the C3D features by your own (Please read the ```sparseprop.feature``` for guidelines on how to format the C3D features.). 

* **Code for training a new model:** Use the script ```run_train.py``` to train a model using a new dataset (or features). For further information, please read the documentation in the script.

# Try our demo!
SparseProp provides a demo that takes as input C3D features from a sample video and a Class-Induced pre-trained model to retrieve temporal segments that are likely to contain human actions. To try our demo, run the following command:

```python retrieve_proposals.py data/demo_input.csv data/demo_c3d.hdf5 data/class_induced_thumos14.pkl data/demo_proposals.csv```

The program above must generate a CSV (data/demo_proposals.csv) file containing the temporal proposals retrieved with an asociated score for each.

# Who is behind it?

| ![Fabian Caba Heilbron][image-cabaf] | ![Juan Carlos Niebles][image-jc] | ![Bernard Ghanem][image-bernard] |
| :---: | :---: | :---: |
| Main contributor | Co-Advisor | Advisor |
| [Fabian Caba][web-cabaf] | [Juan Carlos Niebles][web-jc] | [Bernard Ghanem][web-bernard] |

# Do you want to contribute?

1. Check the open issues or open a new issue to start a discussion around your new idea or the bug you found
2. For the repository and make your changes!
3. Send a pull request


<!--Images-->
[image-cabaf]: http://activity-net.org/challenges/2016/images/fabian.png "Fabian Caba Heilbron"
[image-jc]: http://activity-net.org/images/juan.png "Juan Carlos Niebles"
[image-bernard]: http://activity-net.org/images/bernard.png "Bernard Ghanem"

[image-intro]: https://raw.githubusercontent.com/cabaf/website/gh-pages/temporalproposals/img/pull_figure.png

<!--Links-->
[web-cabaf]: http://www.cabaf.net/
[web-jc]: http://www.niebles.net/
[web-bernard]: http://www.bernardghanem.com/
