"""Allgemeine Funktionen"""

# Erstellt von Simon Henes

# Importieren der benötigten Module
import os
import time


# Funktion um ein neues Run Directory mit der Zeit des Erstellungszeitpunkt
def create_run_directory(output_path):
    # Feststellen der aktuellen Zeit
    current_time = time.strftime("%Y-%b-%d_%H-%M-%S")
    # Erstellen des Namens des Run Diretories
    run_directory_string = "Run_" + current_time

    # Hinzufügen des Run Directories zum output path und erstellen des Directories
    Run_Directory = os.path.join(output_path, run_directory_string)
    os.mkdir(Run_Directory)

    # Zurückgeben des Directory Namens
    return Run_Directory


# Funktion die ein Directory und einen Subdirectory Namen annimmt
# Im übergebenen Directory wir ein neues Subdirectory erstellt, falls dieses noch nicht existiert
def create_subdirectory(directory, subdirectory_name):
    # Erstellen des Subdirectory paths
    sub_dir = os.path.join(directory, subdirectory_name)

    # Erstellen des Directories, falls der path noch nicht existiert
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    # Zurückgeben des Directorypaths
    return sub_dir