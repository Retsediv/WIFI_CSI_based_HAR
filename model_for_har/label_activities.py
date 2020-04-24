from enum import Enum
from os import path

import numpy as np
import pandas as pd

PATH_TO_DATASET = "./dataset/vitalnia_lviv/5"
DATA_FILENAME = "data.csv"
LABEL_FILENAME = "label.csv"


class Activity(Enum):
    NO_PERSON = 0
    STANDING = 1
    WALKING = 2
    SITTING = 3
    LYING = 4
    GET_UP = 5
    GET_DOWN = 6


activity_to_string = {
    Activity.NO_PERSON: "no_person",
    Activity.STANDING: "standing",
    Activity.WALKING: "walking",
    Activity.SITTING: "sitting",
    Activity.LYING: "lying",
    Activity.GET_UP: "get_up",
    Activity.GET_DOWN: "get_down",
}


def set_activity(activities, from_moment, to_moment, activity):
    activities[from_moment:to_moment] = activity_to_string[activity]


def main():
    data = pd.read_csv(path.join(PATH_TO_DATASET, DATA_FILENAME))
    activities = np.empty(shape=(data.shape[0]), dtype=object)

    # Manually defined labels
    set_activity(activities, 0, 230, Activity.STANDING)

    # ...

    activities_df = pd.DataFrame(activities)
    activities_df.to_csv(path.join(PATH_TO_DATASET, LABEL_FILENAME), index=True, header=False)


if __name__ == '__main__':
    main()
