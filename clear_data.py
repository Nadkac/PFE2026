    
import os
import shutil

# Supprime tout le contenu du dossier dataset_cnn
def clear_dataset(self):

    dataset_root = "dataset_cnn"
    
    if os.path.exists(dataset_root):
        # On itère sur tous les fichiers/dossiers dans dataset_cnn
        for filename in os.listdir(dataset_root):
            file_path = os.path.join(dataset_root, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) # Supprime les fichiers
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) # Supprime les dossiers de session
            except Exception as e:
                print(f"Erreur lors de la suppression de {file_path}: {e}")
        print(f"[Maintenance] Dossier {dataset_root} vidé avec succès.")
    else:
        print(f"[Maintenance] Dossier {dataset_root} inexistant.")
