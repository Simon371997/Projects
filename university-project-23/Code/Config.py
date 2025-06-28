"""Definieren der Parameter"""

# Importieren aller Module
import os
import LoadData as ld



# Laden der initalen JSON-Datei mit allen Konfigurationsparametern
parameters = ld.get_json_data("Input/configurations.json")
# Parameter zu den Input-Dateien
INPUT_TEMP_FILE_PATH = parameters["file_paths"]["windows"]["input"]["temp_file_path"]
INPUT_IND_FILE_PATH = parameters["file_paths"]["windows"]["input"]["ind_file_path"]
# Parameter für die Output Dateien
OUTPUT_PATH = parameters["file_paths"]["windows"]["output"]["directory_path"]


# Parameter für die Einrichtung des Models
COUNTRY_TO_USE = parameters["model_parameters"]["country_data_to_use"]
DEPENDENT_VAR = parameters["model_parameters"]["dependent_variable"]