# Human Activity Recognition based on Wi-Fi CSI Data - A Deep Neural Network Approach

This is a repository with source code for the [paper "Human Activity Recognition based on Wi-Fi CSI Data - A Deep Neural Network Approach"](https://www.sciencedirect.com/science/article/pii/S1877050921024509) 
and respective [thesis](https://s3.eu-central-1.amazonaws.com/ucu.edu.ua/wp-content/uploads/sites/8/2021/07/Zhuravchak-Andrii_188586_assignsubmission_file_Bachelor_Thesis_Human_Activity_Recognition_based_on_WiFi_CSI_data.pdf)
(it contains more details that are not covered in the paper).

Using Wi-Fi Channel State Information (CSI) is a novel way of sensing and human activity recognition (HAR). Such a
system can be used in medical institutions for their patients monitoring without privacy violence, as it could be with a
vision-based approach.

The main goal of this thesis was to explore current methods and systems which use Wi-Fi CSI, conduct experiments to
analyze how different hardware configurations affect the data and possibility to detect human activity, collect the
dataset and build the classification model for HAR task. 8 experiments were performed, the dataset in 3 different rooms
was collected, and LSTM-based classification model was build and trained. We’ve shown the full pipeline of building
Wi-Fi CSI based system.

## Repository structure

- `router` - contains source code for `sendData` and `recvCSI`. 
They are used to send data packet from one router and calculate the CSI data on another. 
Then `recvCSI` sends the data to a user computer via UDP connection for further processing.

- `data_retrieval` - contains a program (`run_visualization_server.py`) that listens to `recvCSI` program, 
visualizes incoming data, and saves it to a file. Also, it has a script for a dummy server to emulate incoming data from
the router (`run_test_client.py`) and a sample CSI data in binary format as it is coming from the router 
(`data/sample_csi_packet_big_endian.dat`). 

- `model` - has all the code for building the model and training it, scripts that were used to label activities,
notebook for EDA, etc.


## Dataset

The dataset can be downloaded by the following [link](https://doi.org/10.6084/m9.figshare.14386892.v1).

## Authors

* **[Andrew Zhuravchak](https://github.com/Retsediv)** - Ukrainian Catholic University (UCU) former student
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
