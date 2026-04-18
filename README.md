
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/planche-logo-floracle.jpg?raw=true)

# Définition des objectifs

Lors de l'implantation et du choix de la localisation d'un rucher, un apiculteur, qu'il soit amateur ou professionnel cherchera entre autres à déterminer les plantes dans un rayon d'environ 10 kms (rayon d'action maximum des butineuses) susceptibles de pouvoir fournir du pollen et du nectar, nécessaires au bon développement des colonies, mais aussi à la production de miel.

Une fois cette analyse réalisée, il s'agira pour l'apiculteur d'organiser la vie de la colonie de manière que les pics de population coïncident avec les "miellées", c'est à dire aux moments favorables pendant lesquels les abeilles vont être en mesure de récolter et de stocker de grandes quantités de miel. C'est pendant ces miellées que l'apiculteur posera des hausses sur les ruches, de manière à récolter le miel.

Déterminer les miellées dépend de la météo mais également de la floraison des plantes situées à proximité immédiate de la ruche.

Je suis apiculteur amateur (5 ruches) en Alsace. Mon rucher est disposé à proximité d'arbres fruitiers (en minorité) et de vignes (80% des plantes à proximité immédiate). La vigne, même si elle n'est pas réputée pour cela, est une source importante de nectar et de pollen pour les abeilles, et a fortiori pour mes colonies.

J'ai basé mon projet de machine learning sur cette problématique, avec l'idée de pouvoir réaliser un modèle de prévision de floraison (appelé stade phénologique de floraison) pour la vigne permettant l’optimisation des poses des hausses et donc de la production de miel. 

Le contrôle des ruches s'effectuant toutes les semaines, une précision de 4/5 jours serait souhaitable.

# Etat de l'art

La phénologie viticole conditionne directement la qualité de la récolte, la planification des traitements phytosanitaires, la gestion de l'irrigation et l'adaptation aux aléas climatiques, aussi de nombreuses approches traditionnelles sont-elles employées par les viticulteurs à l’échelle internationale :

**Modèle Spring Warming / Degrés-Jours (GDD – Growing Degree Days)**
Modèle le plus ancien et le plus répandu. Il consiste à effectuer une somme à partir d’une date définie (souvent le 1er janvier) des températures journalières supérieures à un palier dépendant de la plante. (5 ou 10°C en ce qui concerne la vigne)  
*García de Cortázar-Atauri et al. (2009) — Growing Degree Days model, dans OENO One.*


**Modèle GFV (Grapevine Flowering Veraison)**
Dérivé du modèle de Spring Warming il adopte des spécificités et des paramètres spécifiques à la vigne. 
*Parker A.K. et al. (2011) — "General phenological model to characterise the timing of flowering and veraison of Vitis vinifera L.", Australian Journal of Grape and Wine Research, 17(2):206-216*

**Modèles sigmoïdes et non-linéaires**
Le modèle sigmoïde part du même principe que le GDD — accumuler de la chaleur pour prédire un stade — mais au lieu d'une simple addition linéaire, la réponse de la vigne à la température suit une courbe en S : l'effet est faible aux températures basses, maximal dans une zone optimale, puis se réduit à nouveau aux températures très élevées. Concrètement, ça permet de mieux coller à la biologie réelle de la plante, qui ne réagit pas de façon strictement proportionnelle à chaque degré supplémentaire.
*Reis S. et al. (2020) — "Grapevine Phenology in Four Portuguese Wine Regions: Modeling and Predictions", Applied Sciences, 10(11), 3708.*

Le tableau ci-dessous résume les métriques de performances de ces modèles sur la floraison de la vigne : 
| Modèle | RMSE (en jours) | Source |
| ------ | --------------- | ------ |
| GDD (T° base 10°C) | Entre 5 et 10 | Fraga et al. / étude Croatia |
| GFV (Parker) | 5.4 | Parker et al., 2011 |
| Sigmoïde GSM | <7 | Reis et al., 2020 |

Ces approches ont cependant des limites : 
•	Les modèles traditionnels supposent que la température seule suffit. Ils ne capturent pas les interactions entre variables : un site en altitude à 45°N ne se comporte pas comme la somme de l'effet altitude + l'effet latitude séparément.
•	Le modèle GDD se base sur un hyperparamètre de seuil. Pour être précis, il faut calibrer un seuil différent par cépage (Grenache, Pinot Noir, Chardonnay...) et potentiellement par région. Cela nécessite plusieurs années d'observations par site et une expertise agronomique d’où une généralisation très difficile.

L’approche machine learning en la matière permet : 
•	Une meilleure généralisation.
•	Le modèle apprend implicitement que le seuil thermique varie selon la localisation, sans qu'on ait besoin de lui expliquer pourquoi (cépage, microlimat local, exposition).
•	Intégration de variables multiples.
•	L'ajout de nouvelles variables (rayonnement solaire, précipitations, photoperiode) se fait en réentraînant le modèle — pas besoin de reformuler le modèle mathématique.

# Choix techniques et processus

## Choix des modèles
Mon objectif est de pouvoir prédire la date de floraison relative (dans N jours) en fonction de la date et des variables météorologiques courantes. En effet il est important, dans le cadre d’usage défini, de pouvoir suivre la prédiction jour après jour pour que celle-ci s’adapte aux conditions actuelles. Pour effectuer cette prédiction les modèles de régression sont les plus adaptés. 

Je vais ainsi mettre en œuvre et comparer les performances des modèles de régression suivants : 
•	Régression linéaire
•	Régression pénalisée Ridge L2
•	Régression pénalisée Lasso L1
•	Régression pénalisée ElasticNet
•	K-NN (en régression)
•	Arbre de décision (en régression)
•	Forêt aléatoire (en régression)
•	Gradient Boosting

## Evaluation
Je me propose d’évaluer l’efficacité des modèles obtenu en comparant les résultats avec le modèle traditionnel le plus employé, c’est-à-dire GDD, puis de d'appliquer des prévisions sur des cas concrets et de les comparer avec des données non issues des données d'entraînement afin de confirmer les capacités de généralisation.

Les métriques choisies pour l'évaluation sont les suivantes : 
- MAE : Mean Absolute Error : C'est la moyenne des erreurs absolues entre prédiction et valeur réelle. Exprimée dans la mm unité que la variable. (Ici des jours)
- RMSE : Root Mean Squared Error : On élève les erreurs au carré avant de faire la moyenne, puis on prend la racine : Le fait de mettre au carré pénalise fortement les grandes erreurs. Une erreur de 10 jours ne pèse pas 10× mais 100× plus qu'une erreur de 1 jour. Le RMSE s'exprime dans la même unité que la variable (ici, des jours).
- % de prédictions justes à +/- 3 jours
- % de prédictions justes à +/- 7 jours
- Temps d'inférence pour 100 prédictions
- Temps d'entraînement

L’indicateur R² n’a pas été utilisé, n’étant pas adapté à ce cas d’usage. En effet un R² élevé indique que le modèle sait que "plus on est tard dans l'année, moins il reste de jours avant la floraison". C'est vrai, mais trivial. La vraie question — "ce site fleurit-il tôt ou tard par rapport aux autres ?" — est beaucoup plus difficile, et c'est ce que mesurent la MAE et la RMSE.

L'indicateur de RMSE sera privilégié par rapport au MAE. Celui-ci est plus adapté dans un contexte de modèle de phénologie. En effet, se tromper de 2 jours sur la date de floraison, c'est acceptable. Se tromper de 15 jours, c'est potentiellement manquer complètement la période favorable de pose des hausses. Le RMSE, en pénalisant quadratiquement, reflète cette réalité : les grandes erreurs sont disproportionnellement problématiques.

# Données

## Sources de données
La première étape de ma démarche a été d'identifier les typologies de données nécessaires :

**Données phénologiques**
L’INRAE (Institut national de la recherche agronomique) met à disposition en ligne sur sa plateforme TEMPO (*https://tempo.pheno.fr*) des données de constatation de stade phénologiques issues de 10 observatoires, 95 partenaires et 275 contributeurs. L'intégralité des données n'est cependant pas en accès libre, j'ai dû donc effectuer une demande d'accès en tant qu'étudiant réalisant un projet de recherche.

**Données météorologiques**
Les modèles traditionnels se basant sur des données météorologiques il m’a semblé important d’en intégrer aux données d’entraînement. Metéo France propose en accès ouvert des jeux de données complets sur la France entière depuis 1950. 
*https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes/*

## Construction du jeu de données

Données qui nous intéressent issues des données météorologiques : 
- Latitude
- Longitude
- date
- température maximale
- température minimale
- altitude

Les données ne sont pas récupérables chez Météo France par un flux unique mais par un ensemble de flux
Il a donc fallu : 
- Récupérer la liste des urls sources depuis le site Meteo France
- Récupérer les fichiers sources
- Consolider les données dans un fichier parquet
Cf. notebook  :  A.recuperation-data-donnees-meteo.ipynb

Les données phénologiques (dossier data/raw/tempo_202310201126) ont été récupérées depuis le portail de données TEMPO https://tempo.pheno.fr
(Téléchargement direct une fois connecté à la plateforme)
Cf. Notebook : B.exploration-et-consolidation-phenologie.ipynb

# Exploration des données

## Dispersion géographique
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/localisation.png?raw=true)

## Dispersion de la date de floraison
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/dispersion.png?raw=true)
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/distribution.png?raw=true)

## Evolution de la floraison par année
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/evolution-par-annee.png?raw=true)

# Préparation des données

## Démarche
Étapes suivies pour la préparation des données : 
- Chargement des différentes sources de données de phénologies et transformation en un format unifié
- Filtrer pour ne conserver que les données liées à la vigne
- Supprimer les lignes pour lesquelles l'altitude n'est pas renseignée
- Ne conserver que le stade phénologique BBCH 60 (début de floraison)
- Sauvegarde des données réduite dans un fichier (pour reprise facilitée)
- Construction du fichier consolidé comprenant les données de phénologie et de température

## Identification des features à conserver

Les données brutes comprennent les attributs suivants :
- latitude
- longitude
- altitude
- date de floraison
- année

Une analyse de corrélation (test de Pearson) a permis de confirmer la pertinence de ces variables pour prédire date de floraison, même si elle est relativement faible en ce qui concerne l’année et l’altitude.
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/correlation.png?raw=true)

## Feature engineering

Le croisement avec les données météorologiques permettent de calculer les attributs complémentaires suivants :
- temps thermique 10 (GDD10)
- froid hivernal

**Le temps thermique 10 (GDD10)**
Façon de mesurer non pas le temps calendaire, mais le temps biologique d'une plante. L'idée fondatrice est simple : le développement végétal n'est pas piloté par les jours qui passent, mais par la chaleur accumulée au fil du temps. L'indicateur est calculé en additionnant les degrés au dessus de 10 par jour, depuis le 1er janvier de l'année considérée. Pourquoi 10 degrés ? Les bourgeons de vigne ne reprennent leur activité cellulaire active qu'aux alentours de 10 °C. En dessous, les divisions cellulaires sont quasi nulles.

**froid hivernal**
La vigne, comme la plupart des arbres fruitiers des régions tempérées, entre chaque année dans une période de dormance hivernale. Pour en sortir et reprendre son développement au printemps, elle doit au préalable avoir été exposée à un cumul suffisant de températures fraîches — c'est ce qu'on appelle le froid hivernal ou chilling. Sans cette exposition, les bourgeons restent bloqués même si les conditions printanières redeviennent favorables. Il est calculé ici en nombre de jours avec une température moyenne inférieure à 7 degrés, depuis le 1er novembre de l'année précédente.

Le but du modèle est de pouvoir prédire, par rapport à la date du jour, la date de floraison. Ce qui nous intéresse, c'est le jour et le mois de cette floraison, indépendamment de l'année absolue. Aussi ai-je converti la date de floraison en jour N depuis le 1er janvier. Le réchauffement climatique ayant certainement un impact sur l'évolution de la floraison, j'ai jugé bon d'intégrer dans les features l'année, extraite de la date. Soit les deux attributs additionnels ajoutés aux données :
- Jour N de floraison
- Année

L’exploitation souhaitée du modèle me contraint à entraîner le modèle à prédire depuis n'importe quel moment de la saison. Chaque ligne du jeu de données doit répondre à une « question » possible, exemple : « au 23 janvier, avec 12°C de GDD accumulés, dans combien de jours fleurit ce site ?». Ainsi j’ai créé 365 lignes par ligne de données, c'est à dire une entrée par jour de l'année, en calculant les données de température pour chacune de ces dates et en calculant l'attribut N jours avant la floraison en fonction.

Ce qui donne le label suivant : 
- N jours avant floraison

Et les features suivantes : 
- latitude
- longitude
- altitude
- annee
- temps thermique 10 (GDD10)
- froid hivernal

## Pré-processing

Le préprocessing des données intègre une normalisation (Standard Scaler) sur les features du jeu d’entraînement et une séparation train/valid ainsi qu’un échantillonnage. Celui-ci a été réalisé avec GroupShuffleSplit qui me permet d’éviter des fuites de données entre le jeu d’entraînement et de validation dû aux trajectoires de 365 jours créées. (Eviter les mélanges de données d’une même trajectoire dans le jeu d’entraînement et celui de validation) 

# Recherche d'hyperparamètres

## Modèle KNN
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/hyperparametres-knn.png?raw=true)

## Gradient boosting

## Forêt aléatoire

## Arbre de décision


# Evaluation

## Performances des modèles
| Modèle | Inférence (en ms) | Entrainement (en s) | MAE | RMSE | % à +/- 3 j | % à +/- 7 j |
| ------ | ----------------- | ------------------- | --- | ---- | ----------- | ----------- |
| Régr. linéaire | 8 | 2 | 9.348111 | 10.141592 | 27.06 | 53.08 |
| Régr. Ridge L2 | 1 | 2 | 9.348114 | 10.140907 | 27.05 | 53.08 |
| Régr. Lasso L1 | 1 | 2 | 75.418229 | 75.556915 | 2.31 | 5.40 |
| Régr. ElasticNet | 1 | 2 | 26.572929 | 27.193822 | 10.70 | 22.58 |
| KNN | 4 | 7.3 | 5.090163 | 9.020979 | 31.34 | 63.50 |
| Arbre de décision | 1 | 35 | 3.187009 | 9.027631 | 36.32 | 66.16 |
| Forêt aléatoire | 105 | 2581 | 3.413736 | 7.036325 | 38.49 | 71.14 |
| Gradient Boosting | 2 | 1776 | 4.604754 | 6.457499 | 42.76 | 74.35 |

L’indicateur de performance RMSE est le plus pertinent. En effet, se tromper de 6 jours sur la date de floraison, c'est acceptable. Se tromper de 15 jours, c'est potentiellement manquer complètement la période favorable de pose des hausses. Le RMSE, en pénalisant quadratiquement, reflète cette réalité : les grandes erreurs sont disproportionnellement problématiques.

![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/comparaison-mae.png?raw=true)
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/comparaison-rmse.png?raw=true)

L’analyse des performances des modèles permets d’identifier le modèle de Gradient Boosting comme celui avec la plus grande efficacité. De fait, même si la métrique MAE est meilleure sur la forêt aléatoire, l’indicateur de RMSE donne la priorité au Gradient Boosting, or celle-ci est la plus adaptée pour un modèle de phénologie. En effet, se tromper de 2 jours sur la date de floraison, c'est acceptable. Se tromper de 15 jours, c'est potentiellement manquer complètement la période favorable de pose des hausses. Le RMSE, en pénalisant quadratiquement, reflète cette réalité : les grandes erreurs sont disproportionnellement problématiques.

## Analyse des performances du modèle de Gradient Boosting
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/gradientboosting-distribution-erreurs.png?raw=true)
Le modèle présente de bonnes caractéritiques de biai : en moyenne, il ne surestime ni ne sous-estime systématiquement. C'est une propriété intéressante pour un modèle de prédiction phénologique.
La forme ressemble à une cloche centrée autour de 0, ce qui indique un comportement cohérent et sans dérive systématique. La distribution semble légèrement leptokurtique (pic prononcé, queues non négligeables) : le modèle est souvent bon, mais commet parfois des erreurs importantes. Ce n'est pas une distribution strictement gaussienne.

![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/gradientboosting-overfitting.png?raw=true)
L'écart entre les performances en entraînement et validation fait apparaitre un risque d'overfitting modéré:  le modèle a mémorisé une partie du bruit d'entraînement, sans pour autant s'effondrer en validation.
Le RMSE qui grimpe plus fort que la MAE (+40% vs +44% — proches ici) confirme que la validation produit davantage d'erreurs importantes que le train, ce qui est visible sur le troisième graphique, ou pour le dire autrement le modèle gère moins bien les cas rares, ce qui est attendu. Une évolution des modèles pourrait consister à chercher à réduire ce surapprentissage en entraînant les modèles avec davantage de données ou avec d'autres hyperparamètres. 


Afin de valider l’intérêt de la démarche en Machine Learning par rapport aux approches traditionnelles j’ai appliqué la méthode GDD (la plus employée) aux données du jeu de validation et comparé les performances avec le meilleur modèle : 
Voici les performances constatées : 
| Modèle | MAE | RMSE |
| ------ | ----------------- | ------------------- |
| Modèle GDD | 8.13 | 10.32 |
| Gradient Boosting | 4.89 | 6.47 |

Pour mieux appréhender ces résultats il convient de prendre en considération :
•	La variance biologique : sur un même ensemble de plantations des variances de plus de 5 jours peuvent être constatés dans la floraison. Ceci s’explique par des facteurs difficilement mesurables : âge de la vigne, composition des sols, microtopographie…
•	Seuil moyen GDD : le calcul de performance sur la méthode GDD a été effectué avec un hyperparamètre de seuil moyen. Les résultats pourraient donc être meilleurs avec un paramétrage issu d’un historique de données locales à chaque site. Cela démontre cependant l’efficacité de l’usage d’une démarche en Machine Learning dans un contexte de généralisation.


# Cas pratiques

## Application de prédictions sur des données du jeu d'entraînement
![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/exemple-prevision.png?raw=true)
Date moyenne prévue : 30 May 2019 (J150)
Date réelle         : 03 June 2019 (J154)

## Application de prédictions sur des données totalement inconnues
Afin de valider la démarche et la capacité de généralisation j'applique le modèle et je compare la prévision obtenue à des dates réelles constatées par une source non incluse dans les données d'entraînement, en l'occurence le Comité Interprofessionnel des Vins d'Alsace (CIVA) dont les données de constatation de phénologie sont disponibles à l'url suivante : https://technique.vinsalsace.pro/index.php?p=pheno

![alt text](https://github.com/gerard-loic/floracle/blob/master/notebooks/images/exemple-prevision-bergheim-2018.png?raw=true)
Date moyenne prévue : 26 May 2018 (J146)
Date réelle         : 29 May 2018 (J149)

# Application d'exploitation

L'application Flask présente dans le dossier app/ permet de tester des préditions.
Execution : "python3 main.app"

# Critique et amélioration
Une recherche d’optimisation pourrait passer par les améliorations suivantes : 
•	Compléter les données avec des features pouvant impacter la phénologie. (Cépage, taux d’humidité, taux journalier de GDD pour représenter la vitesse de réchauffement, précipitations cumulées, amplitude thermique jour/nuit…)
•	Travailler à un dataset de phénologie couvrant une période temporelle plus large pour une meilleure compréhension de l’impact du changement climatique sur la phénologie.
•	L’approche en Machine Learning a pour inconvénient d’être moins explicable qu’une méthode traditionnelle GDD. L’utilisation d’une librairie comme Shap permettrait de d’expliquer chaque prédiction individuelle.

