import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


TARGET = 'p1_won'


def print_accuracy_for_test(initial_model, train, test, model_title):
    """
    print the accuracy of the model (fitted on the train) on the test.
    :param initial_model: learning model, initiated.
    :param train: train set to fit
    :param test: test set to predict
    :param model_title: model name for readable printing.
    """
    X_train = train.drop([TARGET], axis=1)
    y_train = train[TARGET]
    X_test = test.drop([TARGET], axis=1)
    y_test = test[TARGET]

    initial_model.fit(X_train, y_train)
    test_acc = initial_model.score(X_test, y_test)

    print(f'{model_title}: accuracy: {test_acc}. \n full model: {initial_model}')


def main():
    """
    For each question train dataset, fit a model by the parameters found at tuning, and print the accuracy on the test
    set of this question.
    Print the accuracy of the test set on the models fitted on the train set with the chosen parameters from the tuning.
    for each question:
     * load the train and test set
     * initiate model with the chosen parameters from the tuning
     * fit the model on the train set
     * predict on test set
     *print the accuracy on test set
    """
    # all data models parameters
    train_all_data = pd.read_csv('Data/train/all_years_final_normal_id.csv')
    test_all_data = pd.read_csv('Data/test/test_all_years_final.csv')

    knn_model = KNeighborsClassifier(n_neighbors=75)
    print_accuracy_for_test(knn_model, train_all_data, test_all_data, 'KNN')

    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=11)
    print_accuracy_for_test(dt_model, train_all_data, test_all_data, 'Decision Tree')

    svm_model = LinearSVC(C=10, max_iter=5000)
    print_accuracy_for_test(svm_model, train_all_data, test_all_data, 'SVM')

    svm_model_rbf = SVC(C=100, kernel='rbf', gamma=0.01)
    print_accuracy_for_test(svm_model_rbf, train_all_data, test_all_data, 'SVM RBF kernel')

    svm_model_poly = SVC(C=10, kernel='poly', degree=2)
    print_accuracy_for_test(svm_model_poly, train_all_data, test_all_data, 'SVM Poly kernel')

    random_forest = RandomForestClassifier(criterion='entropy', max_depth=15, max_features='sqrt', min_samples_leaf=1,
                                           min_samples_split=3, n_estimators=320)
    print_accuracy_for_test(random_forest, train_all_data, test_all_data, 'Random Forest')

    adaboost = AdaBoostClassifier(n_estimators=1250)
    print_accuracy_for_test(adaboost, train_all_data, test_all_data, 'AdaBoost')

    perceptron = Perceptron(penalty='l1', eta0=0.0001, max_iter=1500)
    print_accuracy_for_test(perceptron, train_all_data, test_all_data, 'Perceptron')

    mlp = MLPClassifier(solver='adam', max_iter=1000, hidden_layer_sizes=(128,), alpha=0.01, activation='relu')
    print_accuracy_for_test(mlp, train_all_data, test_all_data, 'MultiLayer Perceptron')


    # without scores models parameters
    train_without_scores = pd.read_csv('Data/train/train_without_scores.csv')
    test_without_scores = pd.read_csv('Data/test/test_without_scores.csv')

    knn_model = KNeighborsClassifier(n_neighbors=91)
    print_accuracy_for_test(knn_model, train_without_scores, test_without_scores, 'KNN')

    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=12)
    print_accuracy_for_test(dt_model, train_without_scores, test_without_scores, 'Decision Tree')

    svm_model = LinearSVC(C=2.154 , max_iter=5000)
    print_accuracy_for_test(svm_model, train_without_scores, test_without_scores, 'SVM')
    
    svm_model_rbf = SVC(C=10, kernel='rbf', gamma=0.01)
    print_accuracy_for_test(svm_model_rbf, train_without_scores, test_without_scores, 'SVM RBF kernel')

    svm_model_poly = SVC(C=10, kernel='poly', degree=2)
    print_accuracy_for_test(svm_model_poly, train_without_scores, test_without_scores, 'SVM Poly kernel')
    
    random_forest = RandomForestClassifier(criterion='entropy', max_depth=50, max_features='sqrt', min_samples_leaf=1,
                                           min_samples_split=2, n_estimators=820)
    print_accuracy_for_test(random_forest, train_without_scores, test_without_scores, 'Random Forest')

    adaboost = AdaBoostClassifier(n_estimators=1500)
    print_accuracy_for_test(adaboost, train_without_scores, test_without_scores, 'AdaBoost')

    perceptron = Perceptron(penalty='l2', eta0=0.0006, max_iter=1500)
    print_accuracy_for_test(perceptron, train_without_scores, test_without_scores, 'Perceptron')

    mlp = MLPClassifier(solver='sgd', max_iter=1000, hidden_layer_sizes=(128, 128), alpha=1, activation='tanh')
    print_accuracy_for_test(mlp, train_without_scores, test_without_scores, 'MultiLayer Perceptron')


    # static data models parameters
    train_static_match_data = pd.read_csv('Data/train/train_static_match_data.csv')
    test_static_match_data = pd.read_csv('Data/test/test_static_match_data.csv')

    knn_model = KNeighborsClassifier(n_neighbors=39)
    print_accuracy_for_test(knn_model, train_static_match_data, test_static_match_data, 'KNN')

    dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=9)
    print_accuracy_for_test(dt_model, train_static_match_data, test_static_match_data, 'Decision Tree')

    svm_model = LinearSVC(C=0.0464, max_iter=5000)
    print_accuracy_for_test(svm_model, train_static_match_data, test_static_match_data, 'SVM')

    svm_model_rbf = SVC(C=10, kernel='rbf', gamma=0.01)
    print_accuracy_for_test(svm_model_rbf, train_static_match_data, test_static_match_data, 'SVM RBF kernel')

    svm_model_poly = SVC(C=10, kernel='poly', degree=2)
    print_accuracy_for_test(svm_model_poly, train_static_match_data, test_static_match_data, 'SVM Poly kernel')

    random_forest = RandomForestClassifier(criterion='gini', max_depth=40, max_features='sqrt', min_samples_leaf=1,
                                           min_samples_split=12, n_estimators=350)
    print_accuracy_for_test(random_forest, train_static_match_data, test_static_match_data, 'Random Forest')

    adaboost = AdaBoostClassifier(n_estimators=350)
    print_accuracy_for_test(adaboost, train_static_match_data, test_static_match_data, 'AdaBoost')

    perceptron = Perceptron(penalty='l1', eta0=0.063095, max_iter=1500)
    print_accuracy_for_test(perceptron, train_static_match_data, test_static_match_data, 'Perceptron')

    mlp = MLPClassifier(solver='adam', max_iter=1000, hidden_layer_sizes=(128, 128), alpha=0.1, activation='tanh')
    print_accuracy_for_test(mlp, train_static_match_data, test_static_match_data, 'MultiLayer Perceptron')


if __name__ == '__main__':
    main()
