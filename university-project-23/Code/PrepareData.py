"""Vorbereiten der Daten"""

# Erstellt von Johannes Keßler und Simon Henes

# Importieren der benötigten Module
import pandas as pd
import numpy as np
import os
import GeneralFunctions as gf



# Funktion, die die geladenen Temperatur- und Indikator Daten und einen Ländernamen annimmt
# Für die Indikatordaten wird das gewünschte Land ausgewählt und beide Datensätze werden in pandas dataframes umgewandelt
# Anschließend werden die dataframes an der year-Spalte zusammengeführt und zurückgegegeben
def get_full_dataframe(temp_data, ind_data, country_data_to_use, Run_Directory):
    # Umwandeln der Datensätze in pandas dataframes
    df_temp_data = pd.DataFrame(temp_data)
    # Für die Indikatordaten wird zusätzlich das angegebene Land ausgewählt
    df_ind_data = pd.DataFrame(ind_data[country_data_to_use]["data"])

    # Die Spalten für year und value müssen zum mergen in numerische Datentypen umgewandelt werden
    df_temp_data = df_temp_data.astype({"year": "int64", "value": "float64"})

    # Zusammenführen der beiden dataframes
    df = df_temp_data.merge(df_ind_data, on="year")

    # Umbennen der value-Spalte in temperature
    df = df.rename(columns={"value": "temperature"})

    # Erstellen eines Subdirectorys im Run-Directory und speichern des Dataframes darin
    sub_dir = gf.create_subdirectory(Run_Directory, "Data")
    df.to_csv(os.path.join(sub_dir, "cleaned_data.csv"), index=False)

    # Zurückgeben des zusammengeführten dataframes
    return df
    


# Funktion, die ein pandas dataframe, einen Spaltennamen und einen threshold annimmt
# Ausreißer werden für die angegebene Spalte festgestellt und entfernt
# Anschließend wird das bereinigte Dataframe wieder zurückgegeben
def remove_outliers(input_df, column_name, threshold=3):
    # Berechnen des ersten und dritten Quartils
    Q1 = input_df[column_name].quantile(0.25)
    Q3 = input_df[column_name].quantile(0.75)
    # Berechnen des Interquartilbereichs
    IQR = Q3 - Q1

    # Definieren der upper- und lower-bounds des normalen Bereichs des Datasets
    lower_bound = Q1 - threshold*IQR
    upper_bound = Q3 + threshold*IQR

    # Definieren von Arrays mit den Indizes der Ausreißer
    lower_array = np.where(input_df[column_name]<=lower_bound)[0]
    upper_array = np.where(input_df[column_name]>=upper_bound)[0]

    # Entfernen der Zeilen im Dataset die Ausreißer enthalten
    input_df.drop(index=lower_array, inplace=True)
    input_df.drop(index=upper_array, inplace=True)

    # Zurückgeben des bereinigten Datasets
    return input_df



#Funktion um 0- und NaN-Werte aus einer bestimmten Spalte zu entfernen
def remove_zero_and_nan(input_df, column_name):
    # Filtert alle Werte die = 0 sind
    new_df = input_df[input_df[column_name] !=0]
    # Entfernt alle NaN aus dem Dataset
    new_df = new_df.dropna(subset=[column_name])

    # Gibt Dataset zurück
    return new_df



# Funktion, die ein pandas dataframe und einen Spaltennamen annimmt
# Flats werden für die angegebene Spalte festgestellt und entfernt
# Anschließend wird das bereinigte Dataframe wieder zurückgegeben
def remove_flats(input_df, column_name):
    # Extrahieren der Spaltenwerte
    column_values = input_df[column_name].values
    # Definieren des Maximal- und Minimalwerts
    max_value = max(column_values)
    min_value = min(column_values)
    # Berechnen des Wertebereichs
    value_range = max_value - min_value
    # Berechnen der oberen und unteren Grenze
    # Die obere Grenze ergibt sich durch den Maximalwert minus 0.5% des Wertebereichs
    upper_bound = max_value - value_range * 0.005
    # Die untere Grenze ergibt sich durch den Minimalwert plus 0.5% des Wertebereichs
    lower_bound = min_value + value_range * 0.005
    
    # Festlegung wie viele Werte über und unter den beiden Grenzen liegen
    values_above_upper_bound = (input_df[column_name] > upper_bound).sum()
    values_beneath_lower_bound = (input_df[column_name] < lower_bound).sum()

    # If-Schleife um mögliche Flats aus dem Dataframe zu entfernen
    # Befinden sich mehr als 10% aller Werte über der oberen Grenze, werden diese entfernt
    if values_above_upper_bound > (len(column_values) * 0.1):
        input_df = input_df.drop(input_df[input_df[column_name] > upper_bound].index)
    # Befinden sich weniger als 10% aller Werte unter der unteren Grenze, werden diese entfernt
    if values_beneath_lower_bound > (len(column_values) * 0.1):
        input_df = input_df.drop(input_df[input_df[column_name] < lower_bound].index)

    # Zurückgeben des Dataframes
    return input_df