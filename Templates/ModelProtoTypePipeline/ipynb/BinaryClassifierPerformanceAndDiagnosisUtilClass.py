from package import *
from constants import *


class BinaryClassifierPerformanceAndDiagnosisUtilClass:
    def __init__(self, model=None):
        if model:
            self.model = model
        
    def roc(self, data, target_col_name, prediction_col_name):
        fpr, tpr, _ = roc_curve(data[target_col_name], data[prediction_col_name], pos_label=self.model.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        roc_auc = round(auc(fpr, tpr), 2)
        
        return roc_display, roc_auc

    def precision_recall(self, data, target_col_name, prediction_col_name):
        precision, recall, _ = precision_recall_curve(data[target_col_name], data[prediction_col_name], pos_label=self.model.classes_[1])
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        average_precision = round(average_precision_score(data[target_col_name], data[prediction_col_name]), 2)
        pr_auc = round(auc(recall, precision), 2)

        return pr_display, average_precision, pr_auc

    def roc_pr_curves(self, data, target_col_name, prediction_col_name, model, title, save_to_path=None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        roc_display, roc_auc = self.roc(data, target_col_name, prediction_col_name)
        roc_display.plot(ax=ax1, label=f"AUC {roc_auc} ({name})", linewidth=2)
        pr_display, average_precision, pr_auc = self.precision_recall(data, target_col_name, prediction_col_name, model)
        pr_display.plot(ax=ax2, label=f"AUC:{pr_auc}, AP:{average_precision} ({name})")

        plt.suptitle(title)
        if save_to_path:
            plt.savefig(fname=save_to_path, bbox_inches="tight")
        plt.show()
    
    def roc_pr_curves_train_test(self, df_train, df_test, target_col_name, prediction_col_name, suptitle, save_to_path=None):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for name, data in zip(["train", "test"], [df_train, df_test]):
            roc_display, roc_auc = self.roc(data, target_col_name, prediction_col_name)
            roc_display.plot(ax=ax1, label=f"AUC {roc_auc} ({name})", linewidth=2)
            
            pr_display, average_precision, pr_auc = self.precision_recall(data, target_col_name, prediction_col_name)
            pr_display.plot(ax=ax2, label=f"AUC:{pr_auc}, AP:{average_precision} ({name})")

        fig.suptitle(suptitle)
        plt.grid()
        if save_to_path:
            plt.savefig(fname=save_to_path, bbox_inches="tight")
        plt.show()

    def histogram_of_predicted_prob(self, data, target_col_name, prediction_col_name):
        df_0 = data[data[target_col_name]==0]
        df_1 = data[data[target_col_name]==1]
        plt.hist(df_0[prediction_col_name], color="green", label="class 0")
        plt.hist(df_1[prediction_col_name], color="red", label="class 1")

    def histogram_of_predicted_prob_train_test(self, df_train, df_test, target_col_name, prediction_col_name, suptitle, save_to_path=None):

        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        self.histogram_of_predicted_prob(df_train, target_col_name, prediction_col_name)
        plt.xlabel("predicted class 1 probability")
        plt.ylabel("frequency")
        plt.title("train")
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        self.histogram_of_predicted_prob(df_test, target_col_name, prediction_col_name)
        plt.xlabel("predicted class 1 probability")
        plt.ylabel("frequency")
        plt.title("test")
        plt.grid()
        plt.legend()
        plt.suptitle(suptitle)
        
        if save_to_path:
            plt.savefig(fname=save_to_path, bbox_inches="tight")
        plt.show()

    def learning_curve_n_estimators(self, estimator, eval_data_sets, eval_metrics):
        """
        plot learning curves (loss/model metrics against n_estimators, i.e. number of trees in RF or 
        number of boosting rounds in boosting method), it helps visualize overfit/underfit.

        Parameters
        ----------
        estimator: 
            an untrained XGboost estimator with hyperparameters specified, i.e. XGBClassifier()
        eval_metrics: list[str]
            i.e. ["logloss", "auc", "aucpr"]
        eval_data_sets : list[tuples]
            datasets for evaluation, typically set to be [(X_train, y_train), (X_test, y_test)]

        Returns
        -------
        """
        X_train, y_train = eval_data_sets[0]  
        estimator.fit(X_train, y_train, eval_metric=eval_metrics, eval_set=eval_data_sets, verbose=False)

        # retrieve performance metrics
        results = estimator.evals_result()
        x_axis = range(0, len(results['validation_0'][eval_metrics[0]]))
        n_rows = ceil(len(eval_metrics)/2)
        plt.figure(figsize=(14, n_rows*6))
        for i, eval_metric in enumerate(eval_metrics):
            plt.subplot(n_rows, 2, i+1)
            plt.plot(x_axis, results['validation_0'][eval_metric], label='Train')
            plt.plot(x_axis, results['validation_1'][eval_metric], label='Test')
            plt.legend()
            plt.xlabel("n_estimators")
            plt.ylabel(eval_metric)

        plt.show()

    
    def plot_learning_curve(
        self,
        estimator,
        #title,
        X,
        y,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy"
    ):
        """
        Generate 1 plot: the test and training learning curve

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        _, axes = plt.subplots(1, 1, figsize=(6, 6))

        #axes.set_title(title)
        if ylim is not None:
            axes.set_ylim(*ylim)
        axes.set_xlabel("Training examples")
        axes.set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
            scoring=scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes.grid()
        axes.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes.plot(
            train_sizes, train_scores_mean, "o-", color="r", label=f"Training score ({scoring})"
        )
        axes.plot(
            train_sizes, test_scores_mean, "o-", color="g", label=f"Test score ({scoring})"
        )
        axes.legend(loc="best")

        return plt

    def plot_learning_curve_2(
        self,
        estimator,
        title,
        X,
        y,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
        only_plot_learning_curve=False
    ):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        if axes is None:
            if not only_plot_learning_curve:
                _, axes = plt.subplots(1, 3, figsize=(20, 5))
            else:
                _, axes = plt.subplots(1, 1, figsize=(6, 6))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
            scoring=scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label=f"Training score ({scoring})"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label=f"Test score ({scoring})"
        )
        axes[0].legend(loc="best")

        if not only_plot_learning_curve:
            # Plot n_samples vs fit_times
            axes[1].grid()
            axes[1].plot(train_sizes, fit_times_mean, "o-")
            axes[1].fill_between(
                train_sizes,
                fit_times_mean - fit_times_std,
                fit_times_mean + fit_times_std,
                alpha=0.1,
            )
            axes[1].set_xlabel("Training examples")
            axes[1].set_ylabel("fit_times")
            axes[1].set_title("Scalability of the model")

            # Plot fit_time vs score
            axes[2].grid()
            axes[2].plot(fit_times_mean, test_scores_mean, "o-")
            axes[2].fill_between(
                fit_times_mean,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
            )
            axes[2].set_xlabel("fit_times")
            axes[2].set_ylabel("Score")
            axes[2].set_title("Performance of the model")

        return plt















