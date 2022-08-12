# Human Activity Recognition (HAR) based on WiFi CSI data

This is a repository with source code for thesis "Human Activity Recognition (HAR) based on WiFi CSI data"

Using Wi-Fi Channel State Information (CSI) is a novel way of sensing and human activity recognition (HAR). Such a
system can be used in medical institutions for their patients monitoring without privacy violence, as it could be with a
vision-based approach.

The main goal of this thesis was to explore current methods and systems which use Wi-Fi CSI, conduct experiments to
analyze how different hardware configurations affect the data and possibility to detect human activity, collect the
dataset and build the classification model for HAR task. 8 experiments were performed, the dataset in 3 different rooms
was collected, and LSTM-based classification model was build and trained. We’ve shown the full pipeline of building
Wi-Fi CSI based system.

## Dataset

The dataset can be downloaded by the following [link](https://doi.org/10.6084/m9.figshare.14386892.v1).

## Authors

* **[Andrew Zhuravchak](https://github.com/Retsediv)** - Ukrainian Catholic University(UCU) student
* **Oleh Kapshii** - supervisor

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This repository is a fork from
  the [Atheros-CSI-Tool-UserSpace-APP](https://github.com/NovelSense/Atheros-CSI-Tool-UserSpace-APP)
  which is based on [Atheros-CSI-Tool](https://github.com/xieyaxiongfly/Atheros-CSI-Tool).
* [RF-pose](http://rfpose.csail.mit.edu/) for inspiration for doing this work
* Cypress Semiconductor company, ASR Ukraine team and especially Oleh Kapshii for support and help

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of
conduct, and the process for submitting pull requests to us.
