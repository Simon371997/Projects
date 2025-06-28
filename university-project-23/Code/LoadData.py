"""Laden der Daten"""



# Importieren der benötigten Module
import xml.etree.ElementTree as ET
import json


# Funktion nimmt einen file-path für einen XML-Datensatz an, importiert diesen und gibt ihn als Pandas Dataframe zurück
def get_xml_data(file_path):
    # Parsen der XML-Datei
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Liste für Daten aus XML-Datei
    data_list = []

    # Loop um jedes Element unter 'data' zu durchlaufen
    for element in root.findall('data'):
        # Zwischenspeichern der einzelnen Daten
        year = element.find('year').text
        value = element.find('value').text
        # Hinzufügen eines Dictionaries mit allen Daten zur data_list
        data_list.append({'year': year, 'value': value})

    # Zurückgeben der Daten aus der XML-Datei als Liste mit Dictionaries
    return data_list


# Funktion nimmt einen file-path für einen JSON-Datensatz an, importiert diesen und gibt ihn als Pandas Dataframe zurück
def get_json_data(file_path):
    # Öffnen der JSON-Datei 
    with open(file_path, "r") as file_object:
        # Laden der JSON-Datei in Python als ein Dictionary
        data = json.load(file_object)

    # Zurückgeben der JSON-Datei als Dictionary
    return data