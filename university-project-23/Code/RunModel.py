"""Ausführen des Regressionsmodells"""



# Importieren der benötigten Module
import os
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import GeneralFunctions as gf



# Funktion die ein dataframe und einen string für eine X-Achse annimmt
# Für jede Spalte des dataframes wird ein eigener Lineplot mit der angegebenen x-Achse erstellt
# Die Diagramme werden als Bild im Run_Directory abgespeichert
def create_time_series(df, x_axis, Run_Directory):
    # Erstellen einer Liste der Spaltennamen, die nicht der x-Achse entsprechen
    y_axes = [col for col in df.columns if col != x_axis]

    # Erstellen eines Subdirectories innerhalb des Run Directories, das die später erstellten Linecharts beinhalten soll
    sub_dir = gf.create_subdirectory(Run_Directory, "Zeitreihen")

    # Schleife um durch die Liste mit allen y-Achsen Spaltennamen zu iterieren und jeweils Diagramm zu erstellen
    for column in y_axes:
        # Erstellen des Diagramms
        plt.figure()
        plt.plot(df[x_axis], df[column])
        plt.xlabel(x_axis)
        plt.ylabel(column)
        plt.title(f"Zeitreihe: {column}")
        # Speichern des Diagramms im neu erstellten Subordner im Run Directory
        plt.savefig(os.path.join(sub_dir, f"zeitreihe_{column}.png"))
        plt.close()



# Funktion um beliebig viele lineare Regressionsanalysen für einen Datensatz zu erstellen
# Nimmt ein Input Dataframe an, eine abhängige Variable und 2 Funktionen um das Dataframe zu bereinigen an
# Gibt eine Result-Liste zurück, die Ergebnisparameter zu allen durchgeführten Analysen beinhaltet
def run_linear_regression_model(input_df, dependent_variable, remove_outliers_func, remove_zero_and_nan_func, remove_flats_func):

    # Liste um Ergebnisse der 76 linearen Regressionen zu speichern
    results_list = []

    # For-Schleife, die durch eine Liste mit allen Spaltennamen itieriert außer die Spalten year und der abhängigen Variable
    for column in [col for col in input_df.columns if col != "year" and col != dependent_variable]:

        # Generieren eines neuen Dataframes, das nur die Spalte der abhängigen variable und die aktuelle Indikatorspalte beinhaltet
        model_df = input_df[[dependent_variable, column]].copy()

        # Enterfenen der Ausreißer, Flats und Nullwerte aus der Indikatorspalte
        model_df = remove_outliers_func(model_df, column)
        model_df = remove_zero_and_nan_func(model_df, column)
        model_df = remove_flats_func(model_df, column)

        # Anwendung der linearen Regression auf das Dataframe und Übergabe der unabhängigen und abhängigen Variable
        slope, intercept, r_value, p_value, std_err = stats.linregress(model_df[column], model_df[dependent_variable])

        # Erweitern der Results-Liste um ein Dictionary mit den Ergebnissen
        results_list.append({
            "Indicator": column,
            "Slope": slope,
            "Intercept": intercept,
            "R-Value": r_value,
            "P-Value": p_value,
            "Standard Error": std_err
        })

    # Zurückgeben des Ergebnis-Dictionarys
    return results_list



# Funktion um die Ergebnis-Liste mit den Ergebnis-Dictionarys in einer CSV-Datei zu exportieren
# Nimmt die Ergebnis-Liste und das Run-Directory an
def export_results(results_list, Run_Directory):
    # Umwandeln der Results-Liste in ein pandas Dataframe
    results_df = pd.DataFrame(results_list)

    # Erstellen eines Subdirectorys für die Results
    sub_dir = gf.create_subdirectory(Run_Directory, "Results")

    # Exportieren des Results-Dataframes als CSV im neu erstellten Subordner im Run Directory
    results_df.to_csv(os.path.join(sub_dir, "linear_regression_results.csv"), index=False)

    # Zurückgeben des Results-Dataframes
    return results_df