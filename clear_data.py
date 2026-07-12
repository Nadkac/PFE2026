    
import os
import shutil

# Supprime tout le contenu du dossier dataset_cnn
def clear_dataset_simple():
    
    dataset_root = "dataset_cnn"
    
    print(f"Tentative de nettoyage du dossier : {os.path.abspath(dataset_root)}")
    
    if os.path.exists(dataset_root):
        # On itère sur tout ce qui se trouve dans le dossier dataset_cnn
        for filename in os.listdir(dataset_root):
            file_path = os.path.join(dataset_root, filename)
            try:
                # Si c'est un fichier, on supprime
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Si c'est un dossier , on supprime tout le contenu
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Supprimé : {filename}")
            except Exception as e:
                print(f"Erreur lors de la suppression de {file_path}: {e}")
        print("Nettoyage terminé avec succès.")
    else:
        print(f"Le dossier {dataset_root} n'existe pas, rien à faire.")

if __name__ == "__main__":
    
    confirm = input("Êtes-vous sûr de vouloir supprimer TOUTES les données d'entraînement (y/n) ? ")
    if confirm.lower() == 'y':
        clear_dataset_simple()
    else:
        print("Suppression annulée.")
