
import standardFunc
from standardFunc import timestamp_to_date_utc,date_to_timestamp_utc


import os
import csv
from datetime import datetime
import numpy as np
from numba import njit, prange

if __name__ == "__main__":
    # Exemple de conversion de timestamp en date UTC
    timestamp = 1702335600
  # Exemple de timestamp
    formatted_date = timestamp_to_date_utc(timestamp)
    print("Date UTC formatée:", formatted_date)  # Output: "Date UTC formatée: 2021-06-01 00:02:18"


    # Exemple de conversion de date UTC en timestamp
    year = 2021
    month = 6
    day = 1
    hour = 0
    minute = 2
    second = 18
    timestamp = date_to_timestamp_utc(year, month, day, hour, minute, second)
    print("Timestamp:", timestamp)  # Output: "Timestamp: 1622519738"