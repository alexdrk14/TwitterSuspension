""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Main file that manage the model, feature selection fine-tuning and performance measurements.
####################################################################################################################"""
import sys, argparse
from find_parameters import FineTuning
from visualization import curves

known_features = ["graph_embeddings", "profile", "activity_timing", "post_embeddings", "textual", "combination"]

def plot_curves(gpu, model, verbose):
    global known_features

    """Utilize ids as index, in order to sort legend based on the performance for visualization purposes"""
    ids = [2, 4, 0, 3, 1, 5]
    labels = ["Graph embedding", "Profile", "Activity timing", "Post embeddding", "Textual", "Combination"]
    plot_class = curves()

    for i in ids:
        feature = known_features[i]
        label = labels[i]
        ft_class = FineTuning(gpu=gpu, feature_categ=feature,
                              model_category=model, verbose=verbose,
                              first_portion=True)

        """Load selected features"""
        ft_class.feature_selection(rerun=False)

        """Load model parameters"""
        ft_class.model_class.load_model()

        plot_class.append_new(ft_class.model_class.model, ft_class.X_visible,
                              ft_class.Y_visible, ft_class.X_test,
                              ft_class.Y_test, ft_class.features, label, feature)

        del(ft_class)
    plot_class.plot_pr()
    plot_class.plot_roc()


parser = argparse.ArgumentParser()
parser.add_argument("--finetune", action="store_true", dest='ftune', default=False, help="Parameter fine tuning")
parser.add_argument("--lasso", action="store_true", dest='lasso', default=False,
                    help="Fine tune Lasso feature selection method. Get best alpha param.")
parser.add_argument("--gpu", action="store_true", dest='gpu', default=False,
                    help="Use gpu in XGBoostClassification model")
parser.add_argument("--verbose", action="store_true", dest='verbose', default=False, help="verbose")
parser.add_argument('-f', '--feature', dest="feature", type=str,
                    help="Select the feature category for further usage (select one of:{})".format(known_features))
parser.add_argument("--test", action="store_true", dest='test', default=False,
                    help="Used for measure of test dataset performance.")
parser.add_argument("--plot", action="store_true", dest='plot', default=False,
                    help="Plot SHAP, ROC and Precision vs Recall curves and store them in plots folder.\n\tDo not required specific feature fategory.")

if __name__ == "__main__":
    model_category = "xgboost"
    args = parser.parse_args()

    if not args.plot and args.feature not in known_features:
        print("Choose feature category from known categories: {}".format(known_features))
        sys.exit(-1)

    if args.gpu:
        print("\tGPU use is turned ON")
    else:
        print("\tGPU use is turned OFF")

    if args.verbose:
        print("\tVerbose is ON")
    else:
        print("\tVerbose is OFF")

    if args.plot:
        print("Plot ROC and PR curves...")
        plot_curves(args.gpu, model_category, args.verbose)
    else:
        print("Working on features :{}".format(args.feature))

        if args.lasso:
            print("Fine-tuning Lasso feature selection and store found parameters")

        ft_class = FineTuning(gpu=args.gpu, feature_categ=args.feature, model_category=model_category, verbose=args.verbose,
                              first_portion=False if args.test else True, lasso_ReRun=args.lasso)

        if args.ftune:
            print("Fine-tuning of model with multiple parameters")
            ft_class.model_finetune()
        if args.test:
            print("Measure performance on test data and store ROC and PR curves ")
            ft_class.test_performance()
