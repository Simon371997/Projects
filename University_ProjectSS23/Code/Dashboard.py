

# Importieren aller Module
import Config as c
import LoadData as ld
import PrepareData as pd
import RunModel as rm
import GeneralFunctions as gf

# Laden der Datasets
temp_data = ld.get_xml_data(c.INPUT_TEMP_FILE_PATH)
ind_data = ld.get_json_data(c.INPUT_IND_FILE_PATH)

# Erzeugen des Run Directories
Run_Directory = gf.create_run_directory(c.OUTPUT_PATH)

# Zusammenführen der Datasets in ein pandas dataframe
df = pd.get_full_dataframe(temp_data, ind_data, c.COUNTRY_TO_USE, Run_Directory)

# Erzeugen und Speichern von Zeitreihen für Temperatur und Indikatoren
rm.create_time_series(df, "year", Run_Directory)

# Bereinigen der Ausreißer in den Temperaturwerten
df = pd.remove_outliers(df, c.DEPENDENT_VAR)

# Erstellen der 76 linearen Regressionsanalysen und speichern der Ergebnisse in einer Results-Liste
results_list = rm.run_linear_regression_model(df, c.DEPENDENT_VAR, pd.remove_outliers, pd.remove_zero_and_nan, pd.remove_flats)

# Exportieren der Results-Liste als CSV Datei 
rm.export_results(results_list, Run_Directory)