import numpy as np


def classification_table(y_true, y_pred):
    """
    Generates a classification table of true and false positives and negatives
    for either the binary or multiclass situation.

    Inputs
    ------

    y_true : array of true values for the variable.

    y_pred : array of predicted values for the variable.

    Returns
    ------

    table : classification table of true and false positives and negatives

    Example
    -------

    #Binary Case
    y_true = np.array([0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    print classification_table(y_true, y_pred)

                Predicted

        Actual	0	1

        0	2	1

        1	1	1

    #Multiclass
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 1, 0, 2])
    print classification_table(y_true, y_pred)

            True Pos	False Pos	 True Neg
    0		1		2		4
    1		1		2		4
    2		2		1		5
    """
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)

    if len(classes) > 2:
        table = '\tTrue Pos\tFalse Pos\tTrue Neg\tFalse Neg\n'

        for label in classes:
            true_pos = np.sum(y_pred[y_true == label] == label)
            false_pos = np.sum(y_pred[y_true != label] == label)
            true_neg = np.sum(y_pred[y_true != label] != label)

            table += '{}\t\t{}\t\t{}\t\t{}\n'.format(label, true_pos,
                                                     false_pos, true_neg)

    elif len(classes) == 2:
        true_pos = np.sum(y_pred[y_true == classes[1]] == classes[1])
        false_pos = np.sum(y_pred[y_true != classes[1]] == classes[1])
        true_neg = np.sum(y_pred[y_true != classes[1]] != classes[1])
        false_neg = np.sum(y_pred[y_true == classes[1]] != classes[1])
        table = """
    \tPredicted\n
Actual\t{0}\t{1}\n
{0}\t{2}\t{3}\n
{1}\t{4}\t{5}\n
    """.format(classes[0], classes[1], true_neg, false_pos, false_neg,
               true_pos)
    else:
        raise ValueError('Only one category, results will be meaningless.')

    return table
