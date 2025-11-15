import matplotlib.pyplot as plt
import numpy as np

def create_comparison_table():
    """
    Crée un tableau comparatif entre l'ancienne solution et la nouvelle solution de clustering.
    """
    # Définition des caractéristiques à comparer
    features = [
        "Base de sélection CH", 
        "Prédiction de mobilité", 
        "Support des RSU", 
        "Visualisation", 
        "Logs/Fichiers texte", 
        "Prise en compte trajectoire", 
        "Stabilité des clusters",
        "Performance globale"
    ]
    
    # Évaluation de l'ancienne solution (0-10)
    old_scores = [6, 0, 5, 3, 4, 5, 5, 5]
    
    # Évaluation de la nouvelle solution (0-10)
    new_scores = [9, 10, 10, 9, 10, 9, 8, 9]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Positionnement des barres
    x = np.arange(len(features))
    width = 0.35
    
    # Création des barres
    rects1 = ax.bar(x - width/2, old_scores, width, label='Ancienne Solution', color='lightcoral')
    rects2 = ax.bar(x + width/2, new_scores, width, label='Nouvelle Solution', color='lightgreen')
    
    # Ajout des étiquettes et titre
    ax.set_ylabel('Score (0-10)', fontsize=12)
    ax.set_title('Comparaison entre l\'ancienne et la nouvelle solution de clustering', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    
    # Ajout des valeurs au-dessus des barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('comparison_clustering_solutions.png')
    plt.show()
    
    # Création du tableau textuel détaillé
    print("\n\nTableau comparatif détaillé entre l'ancienne et la nouvelle solution de clustering\n")
    print("=" * 100)
    print("{:<25} | {:<35} | {:<35}".format("Caractéristique", "Ancienne Solution", "Nouvelle Solution"))
    print("=" * 100)
    
    detailed_comparison = [
        ["Base de sélection CH", "Stabilité simple basée sur vitesse et direction", "Score combiné avec prédiction future et historique de mobilité"],
        ["Prédiction de mobilité", "Absente", "Modèle de prédiction basé sur l'historique des positions"],
        ["Support des RSU", "Basique", "Préférence automatique avec zones de couverture configurables"],
        ["Visualisation", "Limitée", "Différentes couleurs selon le rôle (CH: bleu, CM: rouge, RSU-CM: orange)"],
        ["Logs/Fichiers texte", "Fichier unique de suivi", "Multiple fichiers spécialisés (maintenance, sélection CH, clusters RSU)"],
        ["Prise en compte trajectoire", "Direction et vitesse instantanées", "Historique complet avec pondération temporelle et confiance"],
        ["Stabilité des clusters", "Moyenne, changements fréquents de CH", "Améliorée avec bonus de continuité et anticipation des mouvements"],
        ["Performance globale", "Acceptable", "Optimisée avec clustering intelligent basé sur prédiction"]
    ]
    
    for row in detailed_comparison:
        print("{:<25} | {:<35} | {:<35}".format(row[0], row[1], row[2]))
    
    print("=" * 100)
    
    # Sauvegarde du tableau dans un fichier texte
    with open('comparison_clustering_solutions.txt', 'w') as f:
        f.write("Tableau comparatif détaillé entre l'ancienne et la nouvelle solution de clustering\n\n")
        f.write("=" * 100 + "\n")
        f.write("{:<25} | {:<35} | {:<35}\n".format("Caractéristique", "Ancienne Solution", "Nouvelle Solution"))
        f.write("=" * 100 + "\n")
        
        for row in detailed_comparison:
            f.write("{:<25} | {:<35} | {:<35}\n".format(row[0], row[1], row[2]))
        
        f.write("=" * 100 + "\n")

if __name__ == "__main__":
    create_comparison_table()