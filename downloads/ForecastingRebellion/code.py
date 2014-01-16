from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from sklearn import grid_search, svm, preprocessing
from classification_table import classification_table
import pylab as pl

class FitModels(object):
    def __init__(self, x_train, x_test, y_train, y_test, data_name):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.data_name = data_name

    def run_svm(self):
        svm_params = {'C': [2**-4, 2**-3, 2**-2, 2**-1, 2**0,
                        2**1, 2**2, 2**3, 2**4], 
                        'gamma': [0, 2**-4, 2**-3, 2**-2, 2**-1, 2**0,
                        2**1, 2**2, 2**3, 2**4],
                      'kernel': ['linear', 'rbf'],
                      'class_weight': ['auto', None]}
        clf = svm.SVC(probability=True)
        model = grid_search.GridSearchCV(clf, svm_params, n_jobs=3, cv=5)
        print 'Fitting model...'
        model.fit(self.x_train, self.y_train)
        print 'SVM best estimator: {}'.format(model.best_estimator_)
        print 'SVM gridsearchCV scores: {}'.format(model.grid_scores_)
        print 'SVM .score: {}'.format(model.score(self.x_train,self.y_train))
        svm_predicted = model.predict(self.x_test)
        svm_predicted_prob = model.predict_proba(self.x_test)
        print 'SMV precision: {}\n'.format(precision_score(self.y_test,
                                                               svm_predicted))
        print 'SMV recall: {}\n'.format(recall_score(self.y_test,
                                                              svm_predicted))
        print 'SMV f1_score: {}\n'.format(f1_score(self.y_test,
                                                              svm_predicted))

        self.make_roc_plot(self.y_test, svm_predicted_prob, self.data_name, 
                           'SVM')
        class_table_svm = classification_table(self.y_test, svm_predicted)
        filename = 'smv_class_table_{}'.format(self.data_name)
        with open(filename, 'w') as f:
            f.write(class_table_svm)
        

    def run_rf(self):
        rf = RandomForestClassifier(max_depth=None, min_samples_split=1, 
                                    random_state=0)
        rf_parameters = {'n_estimators':[100,250,500,750,1000]}
        rf_grid = grid_search.GridSearchCV(rf, rf_parameters)
        print 'Fitting model...'
        rf_grid.fit(self.x_train,self.y_train)
        print '\nRandom Forest best estimator: {}\n'.format(rf_grid.best_estimator_)
        print 'Random Forest gridsearchCV scores: {}\n'.format(rf_grid.grid_scores_)

        rf_predicted_prob = rf_grid.predict_proba(self.x_test)
        rf_predicted = rf_grid.predict(self.x_test)
        print 'Random forest precision: {}\n'.format(precision_score(self.y_test,
                                                               rf_predicted))
        print 'Random forest recall: {}\n'.format(recall_score(self.y_test,
                                                              rf_predicted))
        print 'Random forest f1_score: {}\n'.format(f1_score(self.y_test,
                                                              rf_predicted))

        self.make_roc_plot(self.y_test, rf_predicted_prob, self.data_name,
                           'Random Forest')
        class_table_rf = classification_table(self.y_test, rf_predicted)
        filename = 'rf_class_table_{}'.format(self.data_name)
        with open(filename, 'w') as f:
            f.write(class_table_rf)

    def run_logit(self):
        lr = LogisticRegression()
        lr_parameters = {'penalty':['l1', 'l2'],
                         'C': [2**-4, 2**-3, 2**-2, 2**-1, 2**0,
                               2**1, 2**2, 2**3, 2**4],
                         'class_weight':['auto', None]}
        lr_grid = grid_search.GridSearchCV(lr, lr_parameters)
        print 'Fitting model...'
        lr_grid.fit(self.x_train,self.y_train)
        print '\nLogistic Regression best estimator: {}\n'.format(lr_grid.best_estimator_)
        print 'Logistic Regression gridsearchCV scores: {}\n'.format(lr_grid.grid_scores_)

        lr_predicted_prob = lr_grid.predict_proba(self.x_test)
        lr_predicted = lr_grid.predict(self.x_test)
        print 'Random forest precision: {}\n'.format(precision_score(self.y_test,
                                                               lr_predicted))
        print 'Random forest recall: {}\n'.format(recall_score(self.y_test,
                                                              lr_predicted))
        print 'Random forest f1_score: {}\n'.format(f1_score(self.y_test,
                                                              lr_predicted))

        self.make_roc_plot(self.y_test, lr_predicted_prob, self.data_name,
                           'Logistic Regression')
        class_table_lr = classification_table(self.y_test, lr_predicted)
        filename = 'lr_class_table_{}'.format(self.data_name)
        with open(filename, 'w') as f:
            f.write(class_table_lr)

    def run_gbt(self):
        bdt = AdaBoostClassifier(DecisionTreeClassifier())
                                                        
        bdt_parameters = {'algorithm':('SAMME', 'SAMME.R'), 
                        'n_estimators':[100,250,500,750,1000]}
        bdt_grid = grid_search.GridSearchCV(bdt, bdt_parameters)
        print 'Fitting model...'
        bdt_grid.fit(self.x_train,self.y_train)
        print 'AdaBoost best estimator: {}\n'.format(bdt_grid.best_estimator_)
        print 'AdaBoost gridsearchCV scores: {}\n'.format(bdt_grid.grid_scores_)

        bdt_predicted_prob = bdt_grid.predict_proba(self.x_test)
        bdt_predicted = bdt_grid.predict(self.x_test)
        print 'AdaBoost precision: {}\n'.format(precision_score(self.y_test,
                                                               bdt_predicted))
        print 'AdaBoost recall: {}\n'.format(recall_score(self.y_test,
                                                              bdt_predicted))
        print 'AdaBoost f1_score: {}\n'.format(f1_score(self.y_test,
                                                              bdt_predicted))

        self.make_roc_plot(self.y_test, bdt_predicted_prob, self.data_name,
                           'AdaBoost')
        class_table_boost = classification_table(self.y_test, bdt_predicted)
        with open('adaboost_class_table.txt', 'w') as f:
            f.write(class_table_boost)

    def run_all(self):
        print '='*15 + '\n'
        print 'Running for data: {}\n'.format(self.data_name)
        print '='*15 + '\n\n'
        print 'Running RF\n\n'
        self.run_rf()
        print 'Running GBT\n\n'
        self.run_gbt()
        print 'Running Logit\n\n'
        self.run_logit()
        print 'Running SVM\n\n'
        self.run_svm()
        print '='*15 + '\n'

    def make_roc_plot(self, test_data, predicted_probs, data_name, model_name):
        """
        Function to generate ROC plots.

        Inputs
        ------

        test_data : y_test data from the train-test splits.

        eoi : The event of interest under examination.

        Output
        ------

        Saves the ROC plot to a file with the EOI in the name.
        """
        fpr, tpr, thresholds = roc_curve(test_data, predicted_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        print 'AUC is {}'.format(roc_auc)

        figname = 'roc_plot_{}_{}'.format(data_name, model_name)
        title = 'ROC Curve - {} - {}'.format(data_name, model_name)
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title(title)
        pl.legend(loc="lower right")
        pl.savefig(figname, bbox_inches=0)

if __name__ == '__main__':
    import numpy as np
    x_train = np.genfromtxt('./x_train.csv', delimiter=',')
    x_test = np.genfromtxt('./x_test.csv', delimiter=',')
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    y_train = np.genfromtxt('./y_train.csv', delimiter=',')
    y_test = np.genfromtxt('./y_test.csv', delimiter=',')
    proc = FitModels(x_train, x_test, y_train, y_test, 'Prediction')
    proc.run_svm()
    #proc.run_all()


    clf = svm.SVC(C=2, cache_size=200, class_weight='auto', coef0=0.0, degree=3,
                  gamma=0, kernel='rbf', max_iter=-1, probability=True, 
                  random_state=None, shrinking=True, tol=0.001, verbose=False)
    clf.fit(x_train, y_train)
    svm_predicted = clf.predict(x_test)
    svm_predicted_prob = clf.predict_proba(x_test)
    print 'SMV precision: {}\n'.format(precision_score(y_test,
                                                            svm_predicted))
    print 'SMV recall: {}\n'.format(recall_score(y_test, svm_predicted))
    print 'SMV f1_score: {}\n'.format(f1_score(y_test, svm_predicted))

    make_roc_plot(y_test, svm_predicted_prob, 'Prediction',  'SVM')
    class_table_svm = classification_table(y_test, svm_predicted)
    filename = 'smv_class_table_{}'.format('Prediction')
    with open(filename, 'w') as f:
        f.write(class_table_svm)


    def make_roc_plot(test_data, predicted_probs, data_name, model_name):
        """
        Function to generate ROC plots.

        Inputs
        ------

        test_data : y_test data from the train-test splits.

        eoi : The event of interest under examination.

        Output
        ------

        Saves the ROC plot to a file with the EOI in the name.
        """
        fpr, tpr, thresholds = roc_curve(test_data, predicted_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        print 'AUC is {}'.format(roc_auc)

        figname = 'roc_plot_{}_{}'.format(data_name, model_name)
        title = 'ROC Curve - {} - {}'.format(data_name, model_name)
        pl.clf()
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title(title)
        pl.legend(loc="lower right")
        pl.savefig(figname, bbox_inches=0)
