import matplotlib.pyplot as plt

# Données
algorithms = ['Régression Logistique', 'KNN', 'SVM', 'Random Forest']
accuracy = [97.66, 97.37, 97.66, 97.08]

# Création du graphique
plt.bar(algorithms, accuracy, color=['blue', 'orange', 'green', 'red'])
plt.xlabel('Algorithmes')
plt.ylabel('Précision (%)')
plt.title('Comparaison des Précisions des Algorithmes')
plt.ylim(96.5, 98)  # Ajuste les limites de l'axe Y pour une meilleure visualisation
plt.show()
