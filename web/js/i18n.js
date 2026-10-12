/* English / French texts. Elements use data-i18n (text) or data-i18n-html (markup). */
(function () {
  "use strict";
  var D = {
    en: {
      navHome: "Home", navAnalyze: "Analyze", navResearch: "Research",
      heroTitle: "Welcome to BugPredictor",
      heroLead: "A tool that predicts which classes of a software project are likely to contain bugs, so that review and testing time goes where it matters.",
      tagline: "Predict the problem before it becomes a problem", start: "Start here",
      f1: "ML-powered", f2: "Bug prediction", f3: "Genetic algorithm",
      howTitle: "How it works",
      how1t: "Code metrics", how1: "Each class is described by 20 metrics: size, complexity, coupling, cohesion, inheritance.",
      how2t: "A default model", how2: "A classifier learns from the classes whose bug history is known.",
      how3t: "Genetic algorithm", how3: "The GA picks the most useful metrics and tunes the model, generation after generation.",
      how4t: "Honest test", how4: "Both models are scored on classes kept aside, which the GA never saw.",
      homeNote: "Pick one of 11 open-source Java projects (PROMISE dataset) or upload your own CSV/ARFF file. In this online demo everything runs in your browser: nothing is sent to a server.",
      homeNoteApi: "Pick one of 11 open-source Java projects (PROMISE dataset) or upload your own CSV/ARFF file. The analysis runs on the local Python server (scikit-learn).",
      anTitle: "Analyze a project", anSub: "Choose the data, choose a model, and let the genetic algorithm try to improve it.",
      s1: "Choose the data", tabPromise: "PROMISE projects", tabUpload: "Upload your file",
      dsHint: "One release of each Apache or open-source Java project; one row per class, bug counts from the issue tracker.",
      dsCard: "{n} classes · {p} defective",
      dropText: "Click to upload or drag and drop", dropFormat: "CSV or ARFF: one row per module, numeric metrics and a bug column",
      uploadHint: "The target column is found automatically (<code>bug</code>, <code>defects</code>, <code>label</code>…, otherwise the last column). Counts above 0, <code>true</code> or <code>yes</code> mean defective. A <code>loc</code> column enables the effort-aware score.",
      fileMeta: "{n} modules · {f} numeric metrics · {d} defective ({p}) · target “{t}”",
      s2: "Choose the model", gens: "Generations", pop: "Population", seed: "Random seed",
      engineBrowser: "Runs in your browser (a JavaScript port of the Python code). SVM and random forest need the Python version: see the README on GitHub.",
      engineApi: "Runs on the local Python server with scikit-learn.",
      pythonOnly: "Python version only",
      startAnalysis: "Start analysis", pickData: "Choose a project or upload a file first.",
      m_knn: "K-nearest neighbours", m_knn_d: "The model of the original project", m_lr: "Logistic regression", m_lr_d: "Linear, fast, often the best baseline",
      m_nb: "Gaussian naive Bayes", m_nb_d: "Very fast, a classic in defect prediction", m_svm: "SVM (RBF kernel)", m_svm_d: "The second model of the project",
      m_rf: "Random forest", m_rf_d: "Trees; slower to optimise",
      st1: "Processing dataset", st2: "Training the default model", st3: "Optimizing with the GA", st4: "Analysis complete",
      pg_processing: "Splitting the modules: 70 % for training, 30 % kept aside", pg_default: "Training the default model",
      pg_ga: "Generation {g} of {n}: best F1 on the inner folds {b}", pg_ga0: "Generation 0: evaluating the first population",
      pg_final: "Scoring the optimized model on the held-out modules", pg_done: "Analysis complete! Results are ready to review.",
      pct: "{p} % complete", cancel: "Cancel", liveBest: "best individual", liveMean: "population mean",
      progMeta: "{model} · {gens} generations × {pop} individuals",
      again: "Run again (new seed)", newAnalysis: "New analysis",
      resTitle: "{ds} · {model}",
      chipSplit: "{tr} training / {te} held-out modules ({d} defective)", chipEngine: "Engine: {e}", chipTime: "{s} s", chipEvals: "{n} models evaluated by the GA",
      engBrowser: "browser (JavaScript)", engPython: "Python (scikit-learn)",
      vGood: "On the {n} held-out modules, the optimized model reaches F1 {o} against {d} for the default model ({diff}), with {k} of {m} metrics.",
      vBad: "On the {n} held-out modules, the default model does better (F1 {d} against {o}). The GA found settings that looked better on its training folds but do not generalise here: try another seed or more generations, and see the Research page.",
      vSame: "On the {n} held-out modules, both models perform about the same (F1 {o} against {d} for the default model); the optimized one needs only {k} of {m} metrics.",
      defModel: "Default model", defSub: "Standard settings, all metrics", optModel: "Optimized model", optSub: "Metrics and parameters chosen by the GA",
      recommended: "Recommended", notBetter: "Not better here",
      perf: "Performance on held-out modules", params: "Model parameters",
      cmTP: "Bugs found", cmFN: "Bugs missed", cmFP: "False alarms", cmTN: "Correctly clean",
      convTitle: "How the genetic algorithm progressed", convSub: "Fitness = mean F1 over 5 inner folds of the training modules (minus a small cost per metric kept).",
      convBest: "best fitness", convMean: "population mean", convF1: "F1 of the best (inner folds)", generation: "generation",
      featTitle: "Metrics kept by the GA", featSub: "{k} of {m} metrics kept. Hover a metric for its meaning.",
      rankTitle: "Classes to review first", rankSub: "Risk from out-of-fold predictions: each class is scored by a model trained without it.",
      downloadCsv: "Download all (CSV)", thRank: "#", thModule: "Class", thLoc: "Lines", thRisk: "Predicted risk", thActual: "Had bugs?",
      yes: "yes", no: "no",
      effort: "Reviewing classes by predicted risk per line until 20 % of the code is read finds <b>{m}</b> of the defective classes. Reviewing the smallest classes first, a simple baseline, finds {b}.",
      noLoc: "No <code>loc</code> column: the effort-aware score is not available.",
      errPrefix: "The analysis failed: ",
      footText: "BugPredictor · final-year project by <a href=\"https://yassine-portfolio-virid.vercel.app\" target=\"_blank\" rel=\"noreferrer\">Yassine Abderrezak Dache</a> · data: PROMISE repository (CK metrics)",
      footCode: "Source code",
      metric_f1: "F1-score", metric_precision: "Precision", metric_recall: "Recall", metric_mcc: "MCC", metric_auc: "AUC",
      // research page
      rTitle: "What the study found",
      rLead: "The models were evaluated on 11 releases of open-source Java projects from the PROMISE repository, with nested cross-validation: the genetic algorithm only ever sees the training part of each fold.",
      rLeakTitle: "1. The first version of the project leaked test data",
      rLeakText: "The original notebooks applied SMOTE (synthetic over-sampling) to the whole dataset before cross-validation. Synthetic copies of each test class then sit in the training folds, and a 1-nearest-neighbour model simply finds them. Re-running both protocols on the same releases shows the size of the bias:",
      rKpiLeaky: "F1 reported by the original protocol", rKpiHonest: "F1 of the same GA, evaluated correctly", rKpiK1: "1-NN: F1 with SMOTE before CV", rKpiK1h: "1-NN: F1 with SMOTE inside each fold",
      rAvg: "mean over 11 releases",
      rLeakChart: "Original protocol vs leakage-free evaluation (k-NN + GA)", sLeaky: "original protocol (leaky)", sHonestGA: "GA, nested CV", sHonestDef: "default k-NN",
      rGaTitle: "2. What the genetic algorithm really brings",
      rGaText: "With an honest evaluation, the GA's gains in F1 are small and depend on the dataset; its clearest benefit is a much smaller model, which keeps about a third of the metrics. The metrics it keeps also change from one run to the next: they are strongly correlated, so many subsets work about as well.",
      thModel: "Model", thDefF1: "Default F1", thGaF1: "GA F1", thWins: "GA better on", thFeat: "Metrics kept", thInner: "GA's own estimate (inner CV)",
      rInnerNote: "The last column is the F1 the GA believed it had reached on its training folds: the gap with the real score is the optimism that nested cross-validation removes.",
      rBaseTitle: "3. Default models, all 11 releases",
      rBaseText: "5 × 5-fold cross-validation, SMOTE and preprocessing fitted inside each training fold.",
      rTransferTitle: "4. Predicting a new release or a new project",
      rTransferText: "Within-project cross-validation is optimistic about real use. Training on the previous release (cross-version) or on the 10 other projects (cross-project) gives:",
      thWithin: "Within project (CV)", thCV: "Previous release", thCP: "Other projects",
      rMore: "Full tables, figures and the code that produced them: <a href=\"https://github.com/yassinedache/bugpredictor/blob/main/results/results.md\" target=\"_blank\" rel=\"noreferrer\">results/results.md</a> on GitHub.",
      rManual: "Recall@20%LOC: share of the defective classes found by reading 20 % of the code, riskiest per line first. Reading the smallest classes first, with no model at all, finds {v}.",
      rPending: "The research results are not available in this build."
    },
    fr: {
      navHome: "Accueil", navAnalyze: "Analyser", navResearch: "Étude",
      heroTitle: "Bienvenue sur BugPredictor",
      heroLead: "Un outil qui prédit quelles classes d'un projet logiciel risquent de contenir des bugs, pour concentrer la relecture et les tests là où c'est utile.",
      tagline: "Prédire le problème avant qu'il ne devienne un problème", start: "Commencer",
      f1: "Apprentissage automatique", f2: "Prédiction de bugs", f3: "Algorithme génétique",
      howTitle: "Comment ça marche",
      how1t: "Métriques de code", how1: "Chaque classe est décrite par 20 métriques : taille, complexité, couplage, cohésion, héritage.",
      how2t: "Un modèle par défaut", how2: "Un classifieur apprend à partir des classes dont l'historique de bugs est connu.",
      how3t: "Algorithme génétique", how3: "L'AG choisit les métriques les plus utiles et règle le modèle, génération après génération.",
      how4t: "Test honnête", how4: "Les deux modèles sont évalués sur des classes mises de côté, que l'AG n'a jamais vues.",
      homeNote: "Choisissez l'un des 11 projets Java open source (jeu de données PROMISE) ou importez votre propre fichier CSV/ARFF. Dans cette démo en ligne, tout s'exécute dans votre navigateur : rien n'est envoyé à un serveur.",
      homeNoteApi: "Choisissez l'un des 11 projets Java open source (jeu de données PROMISE) ou importez votre propre fichier CSV/ARFF. L'analyse s'exécute sur le serveur Python local (scikit-learn).",
      anTitle: "Analyser un projet", anSub: "Choisissez les données et un modèle, puis laissez l'algorithme génétique essayer de l'améliorer.",
      s1: "Choisir les données", tabPromise: "Projets PROMISE", tabUpload: "Importer un fichier",
      dsHint: "Une version de chaque projet Java d'Apache ou open source ; une ligne par classe, nombre de bugs issu du gestionnaire de tickets.",
      dsCard: "{n} classes · {p} défectueuses",
      dropText: "Cliquez pour importer ou glissez-déposez", dropFormat: "CSV ou ARFF : une ligne par module, des métriques numériques et une colonne de bugs",
      uploadHint: "La colonne cible est détectée automatiquement (<code>bug</code>, <code>defects</code>, <code>label</code>…, sinon la dernière colonne). Un nombre supérieur à 0, <code>true</code> ou <code>yes</code> signifie défectueux. Une colonne <code>loc</code> active le score tenant compte de l'effort.",
      fileMeta: "{n} modules · {f} métriques numériques · {d} défectueux ({p}) · cible « {t} »",
      s2: "Choisir le modèle", gens: "Générations", pop: "Population", seed: "Graine aléatoire",
      engineBrowser: "S'exécute dans votre navigateur (portage JavaScript du code Python). Le SVM et la forêt aléatoire nécessitent la version Python : voir le README sur GitHub.",
      engineApi: "S'exécute sur le serveur Python local avec scikit-learn.",
      pythonOnly: "Version Python uniquement",
      startAnalysis: "Lancer l'analyse", pickData: "Choisissez d'abord un projet ou importez un fichier.",
      m_knn: "K plus proches voisins", m_knn_d: "Le modèle du projet d'origine", m_lr: "Régression logistique", m_lr_d: "Linéaire, rapide, souvent la meilleure référence",
      m_nb: "Bayes naïf gaussien", m_nb_d: "Très rapide, un classique de la prédiction de défauts", m_svm: "SVM (noyau RBF)", m_svm_d: "Le second modèle du projet",
      m_rf: "Forêt aléatoire", m_rf_d: "Arbres ; plus lente à optimiser",
      st1: "Préparation des données", st2: "Entraînement du modèle par défaut", st3: "Optimisation par l'AG", st4: "Analyse terminée",
      pg_processing: "Séparation des modules : 70 % pour l'entraînement, 30 % mis de côté", pg_default: "Entraînement du modèle par défaut",
      pg_ga: "Génération {g} sur {n} : meilleur F1 sur les plis internes {b}", pg_ga0: "Génération 0 : évaluation de la première population",
      pg_final: "Évaluation du modèle optimisé sur les modules mis de côté", pg_done: "Analyse terminée ! Les résultats sont prêts.",
      pct: "{p} % effectués", cancel: "Annuler", liveBest: "meilleur individu", liveMean: "moyenne de la population",
      progMeta: "{model} · {gens} générations × {pop} individus",
      again: "Relancer (nouvelle graine)", newAnalysis: "Nouvelle analyse",
      resTitle: "{ds} · {model}",
      chipSplit: "{tr} modules d'entraînement / {te} mis de côté ({d} défectueux)", chipEngine: "Moteur : {e}", chipTime: "{s} s", chipEvals: "{n} modèles évalués par l'AG",
      engBrowser: "navigateur (JavaScript)", engPython: "Python (scikit-learn)",
      vGood: "Sur les {n} modules mis de côté, le modèle optimisé atteint un F1 de {o} contre {d} pour le modèle par défaut ({diff}), avec {k} métriques sur {m}.",
      vBad: "Sur les {n} modules mis de côté, le modèle par défaut fait mieux (F1 {d} contre {o}). L'AG a trouvé des réglages qui semblaient meilleurs sur ses plis d'entraînement mais ne se généralisent pas ici : essayez une autre graine ou plus de générations, et consultez la page Étude.",
      vSame: "Sur les {n} modules mis de côté, les deux modèles font à peu près jeu égal (F1 {o} contre {d} pour le modèle par défaut) ; le modèle optimisé n'utilise que {k} métriques sur {m}.",
      defModel: "Modèle par défaut", defSub: "Réglages standard, toutes les métriques", optModel: "Modèle optimisé", optSub: "Métriques et paramètres choisis par l'AG",
      recommended: "Recommandé", notBetter: "Pas meilleur ici",
      perf: "Performance sur les modules mis de côté", params: "Paramètres du modèle",
      cmTP: "Bugs trouvés", cmFN: "Bugs manqués", cmFP: "Fausses alertes", cmTN: "Sains bien classés",
      convTitle: "La progression de l'algorithme génétique", convSub: "Fitness = F1 moyen sur 5 plis internes des modules d'entraînement (moins un petit coût par métrique conservée).",
      convBest: "meilleure fitness", convMean: "moyenne de la population", convF1: "F1 du meilleur (plis internes)", generation: "génération",
      featTitle: "Métriques conservées par l'AG", featSub: "{k} métriques sur {m} conservées. Survolez une métrique pour sa signification.",
      rankTitle: "Classes à relire en priorité", rankSub: "Risque issu de prédictions hors-pli : chaque classe est évaluée par un modèle entraîné sans elle.",
      downloadCsv: "Tout télécharger (CSV)", thRank: "#", thModule: "Classe", thLoc: "Lignes", thRisk: "Risque prédit", thActual: "Bugs ?",
      yes: "oui", no: "non",
      effort: "En relisant les classes par risque prédit par ligne jusqu'à 20 % du code, on trouve <b>{m}</b> des classes défectueuses. Relire d'abord les plus petites classes, une référence simple, en trouve {b}.",
      noLoc: "Pas de colonne <code>loc</code> : le score tenant compte de l'effort n'est pas disponible.",
      errPrefix: "L'analyse a échoué : ",
      footText: "BugPredictor · projet de fin d'études de <a href=\"https://yassine-portfolio-virid.vercel.app\" target=\"_blank\" rel=\"noreferrer\">Yassine Abderrezak Dache</a> · données : dépôt PROMISE (métriques CK)",
      footCode: "Code source",
      metric_f1: "F1-score", metric_precision: "Précision", metric_recall: "Rappel", metric_mcc: "MCC", metric_auc: "AUC",
      rTitle: "Ce que montre l'étude",
      rLead: "Les modèles ont été évalués sur 11 versions de projets Java open source du dépôt PROMISE, par validation croisée imbriquée : l'algorithme génétique ne voit jamais que la partie entraînement de chaque pli.",
      rLeakTitle: "1. La première version du projet laissait fuiter les données de test",
      rLeakText: "Les notebooks d'origine appliquaient SMOTE (sur-échantillonnage synthétique) à tout le jeu de données avant la validation croisée. Des copies synthétiques de chaque classe de test se retrouvent alors dans les plis d'entraînement, et un modèle du plus proche voisin les retrouve simplement. Relancer les deux protocoles sur les mêmes versions montre l'ampleur du biais :",
      rKpiLeaky: "F1 annoncé par le protocole d'origine", rKpiHonest: "F1 du même AG, évalué correctement", rKpiK1: "1-NN : F1 avec SMOTE avant la VC", rKpiK1h: "1-NN : F1 avec SMOTE dans chaque pli",
      rAvg: "moyenne sur 11 versions",
      rLeakChart: "Protocole d'origine contre évaluation sans fuite (k-NN + AG)", sLeaky: "protocole d'origine (avec fuite)", sHonestGA: "AG, VC imbriquée", sHonestDef: "k-NN par défaut",
      rGaTitle: "2. Ce que l'algorithme génétique apporte vraiment",
      rGaText: "Avec une évaluation honnête, les gains de l'AG en F1 sont faibles et dépendent du jeu de données ; son apport le plus net est un modèle bien plus petit, qui ne garde qu'environ un tiers des métriques. Les métriques gardées changent aussi d'une exécution à l'autre : elles sont fortement corrélées, si bien que beaucoup de sous-ensembles fonctionnent à peu près aussi bien.",
      thModel: "Modèle", thDefF1: "F1 par défaut", thGaF1: "F1 avec l'AG", thWins: "AG meilleur sur", thFeat: "Métriques gardées", thInner: "Estimation de l'AG (VC interne)",
      rInnerNote: "La dernière colonne est le F1 que l'AG pensait atteindre sur ses plis d'entraînement : l'écart avec le score réel est l'optimisme que la validation croisée imbriquée supprime.",
      rBaseTitle: "3. Modèles par défaut, les 11 versions",
      rBaseText: "Validation croisée 5 × 5 plis, SMOTE et prétraitement ajustés dans chaque pli d'entraînement.",
      rTransferTitle: "4. Prédire une nouvelle version ou un nouveau projet",
      rTransferText: "La validation croisée intra-projet est optimiste par rapport à l'usage réel. Entraîner sur la version précédente (inter-versions) ou sur les 10 autres projets (inter-projets) donne :",
      thWithin: "Intra-projet (VC)", thCV: "Version précédente", thCP: "Autres projets",
      rMore: "Tableaux complets, figures et code qui les produit : <a href=\"https://github.com/yassinedache/bugpredictor/blob/main/results/results.md\" target=\"_blank\" rel=\"noreferrer\">results/results.md</a> sur GitHub.",
      rManual: "Recall@20%LOC : part des classes défectueuses trouvées en lisant 20 % du code, les plus risquées par ligne d'abord. Lire d'abord les plus petites classes, sans aucun modèle, en trouve {v}.",
      rPending: "Les résultats de l'étude ne sont pas disponibles dans cette version."
    }
  };
  var GLOSS = {
    en: { wmc: "Weighted methods per class", dit: "Depth of inheritance tree", noc: "Number of children (subclasses)", cbo: "Coupling between objects",
          rfc: "Response for a class", lcom: "Lack of cohesion in methods", ca: "Afferent coupling (classes using this one)", ce: "Efferent coupling (classes this one uses)",
          npm: "Number of public methods", lcom3: "Lack of cohesion (Henderson-Sellers)", loc: "Lines of code", dam: "Data access metric (private attributes)",
          moa: "Measure of aggregation", mfa: "Measure of functional abstraction (inherited methods)", cam: "Cohesion among methods", ic: "Inheritance coupling",
          cbm: "Coupling between methods", amc: "Average method complexity", max_cc: "Maximum cyclomatic complexity", avg_cc: "Average cyclomatic complexity" },
    fr: { wmc: "Méthodes pondérées par classe", dit: "Profondeur dans l'arbre d'héritage", noc: "Nombre de sous-classes", cbo: "Couplage entre objets",
          rfc: "Réponse d'une classe", lcom: "Manque de cohésion des méthodes", ca: "Couplage afférent (classes qui l'utilisent)", ce: "Couplage efférent (classes qu'elle utilise)",
          npm: "Nombre de méthodes publiques", lcom3: "Manque de cohésion (Henderson-Sellers)", loc: "Lignes de code", dam: "Accès aux données (attributs privés)",
          moa: "Mesure d'agrégation", mfa: "Abstraction fonctionnelle (méthodes héritées)", cam: "Cohésion entre méthodes", ic: "Couplage d'héritage",
          cbm: "Couplage entre méthodes", amc: "Complexité moyenne des méthodes", max_cc: "Complexité cyclomatique maximale", avg_cc: "Complexité cyclomatique moyenne" }
  };

  function detect() {
    var q = /[?&]lang=(en|fr)\b/i.exec(location.search || "");
    if (q) return q[1].toLowerCase();
    try { var s = localStorage.getItem("portfolio-lang"); if (s === "en" || s === "fr") return s; } catch (e) {}
    return /^fr/i.test(navigator.language || "") ? "fr" : "en";
  }
  var I = window.I18N = { lang: detect(), listeners: [] };
  I.t = function (key, vars) {
    var s = (D[I.lang] && D[I.lang][key]) || D.en[key] || key;
    if (vars) s = s.replace(/\{(\w+)\}/g, function (m, k) { return vars[k] != null ? vars[k] : m; });
    return s;
  };
  I.gloss = function (f) { return (GLOSS[I.lang] || GLOSS.en)[String(f).toLowerCase()] || ""; };
  I.apply = function (root) {
    root = root || document;
    root.querySelectorAll("[data-i18n]").forEach(function (el) { el.textContent = I.t(el.getAttribute("data-i18n")); });
    root.querySelectorAll("[data-i18n-html]").forEach(function (el) { el.innerHTML = I.t(el.getAttribute("data-i18n-html")); });
    document.documentElement.lang = I.lang;
    document.querySelectorAll("[data-lang]").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-lang") === I.lang ? "true" : "false"); });
  };
  I.set = function (l) {
    I.lang = l;
    try { localStorage.setItem("portfolio-lang", l); } catch (e) {}
    I.apply();
    I.listeners.forEach(function (f) { f(l); });
  };
  I.on = function (f) { I.listeners.push(f); };
  I.num = function (x, d) {
    if (x == null || !isFinite(x)) return "–";
    var s = Number(x).toFixed(d == null ? 2 : d);
    return I.lang === "fr" ? s.replace(".", ",") : s;
  };
  I.pct = function (x, d) { return x == null || !isFinite(x) ? "–" : I.num(100 * x, d == null ? 0 : d) + (I.lang === "fr" ? " %" : "%"); };
  I.int = function (n) { return Math.round(n).toLocaleString(I.lang === "fr" ? "fr-FR" : "en-US"); };
})();
