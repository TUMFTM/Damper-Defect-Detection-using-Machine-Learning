import tensorflow as tf
import math
import sklearn.metrics as slmetrics
import numpy as np
import os


def classfiy(conf, dataset):
    def feed_dict(batchsize=conf.batch_size):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        xs, ys = dataset.test.next_batch(batchsize, shuffle=False)  # disable shuffle
        k = 1.0     # do not use dropout for test
        tr = False
        # Put the data into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        return {x: xs, y_true: ys, keep_prob: k, is_training_phase: tr}

    tf.reset_default_graph()
    with tf.Session() as sess:

        # print the used parameter setup
        for arg in vars(conf):
            print(arg, getattr(conf, arg))

        # load meta graph and restore weights
        saver = tf.train.import_meta_graph(conf.eval_modelpath + 'metagraph.meta')
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(conf.eval_modelpath)))
        # if this (^) fails, delete the path in checkpoint file (the one without file extension)
        graph = tf.get_default_graph()

        # get placeholders
        x = graph.get_tensor_by_name("CNN_Model/Inputs/x:0")
        y_true = graph.get_tensor_by_name("CNN_Model/Inputs/y_true:0")
        is_training_phase = graph.get_tensor_by_name("CNN_Model/Inputs/is_training_phase:0")
        keep_prob = graph.get_tensor_by_name("CNN_Model/Inputs/keep_prob:0")

        if "LeNet" in conf.eval_modelpath:
            # get output and eval op for LeNet
            output = graph.get_tensor_by_name("CNN_Model/logits/BiasAdd:0")
            eval_op = graph.get_tensor_by_name("CNN_Model/Accuracy/Mean:0")
        elif "VGG" in conf.eval_modelpath:
            # get output and eval op for VGGNet
            output = graph.get_tensor_by_name("CNN_Model/logits/BiasAdd:0")
            eval_op = graph.get_tensor_by_name("CNN_Model/Accuracy/Mean:0")
        elif "Verstraete" in conf.eval_modelpath and conf.input == "2D":
            # get output and eval op for VGGNet
            output = graph.get_tensor_by_name("CNN_Model/FC-Block/Logits/BiasAdd:0")
            eval_op = graph.get_tensor_by_name("CNN_Model/Accuracy/Mean:0")
        elif "Verstraete" in conf.eval_modelpath and conf.input == "1D":
            # get output and eval op for VGGNet
            output = graph.get_tensor_by_name("CNN_Model/FC-Block/Output/BiasAdd:0")
            eval_op = graph.get_tensor_by_name("CNN_Model/Accuracy/Mean:0")
        else:
            # get output and eval op
            output = graph.get_tensor_by_name("CNN_Model/Output/BiasAdd:0")
            eval_op = graph.get_tensor_by_name("CNN_Model/Accuracy/Mean:0")



        # test set evaluation, note that the output layer is BEFORE the softmax
        # no shuffling, in last mini batch only use the remaining few samples
        testacc = 0.
        test_predictions = []
        total_batch_test = int(
            math.ceil(dataset.test.num_examples / conf.batch_size))  # make sure to include fractional mini-batch
        remainder = dataset.test.num_examples % conf.batch_size  # how many samples are in the incomplete mini-batch?
        for i in range(total_batch_test):  # loop over all mini-batches
            if i == total_batch_test - 1 and remainder != 0:  # in last of all mini-batches and if there is a remainder
                batch_testacc, test_pred, score = sess.run([eval_op, tf.argmax(tf.nn.softmax(output), axis=1), tf.nn.softmax(output)],
                                                    feed_dict=feed_dict(remainder))  # use reduced mini-batch size
            else:
                batch_testacc, test_pred, score = sess.run([eval_op, tf.argmax(tf.nn.softmax(output), axis=1), tf.nn.softmax(output)],
                                                    feed_dict=feed_dict())  # use full mini-batch size
            testacc += batch_testacc / total_batch_test  # calculate test accuracy over all the mini-batches
            test_predictions.extend(test_pred)  # merge predictions into one list
            if i == 0:
                scores = score
            else:
                scores = np.append(scores, score, axis=0)


        confmat = evaluate_metrics(dataset, test_predictions, conf.metrics_avg)
        print("Accuracy:\t%.4f" % testacc)

        if conf.evaluate and conf.write_file:
            np.savetxt(conf.logspath + 'PredictionScores.csv', scores, delimiter=",")
            np.savetxt(conf.logspath + 'PredictionLabels.csv', test_predictions, fmt='%d')
            np.savetxt(conf.logspath + 'Index.csv', dataset.test.index, fmt='%d')

    return testacc, confmat


def evaluate_metrics(dataset, prediction, average, print_it=True):
    """calculate confusion matrix, recall, precision, f1 score"""
    # get classes instead of one-hot encoding
    dataset.test.cls = np.argmax(dataset.test.labels, axis=1)
    # calculate metrics
    confmat = slmetrics.confusion_matrix(y_true=dataset.test.cls, y_pred=prediction)
    recall = slmetrics.recall_score(y_true=dataset.test.cls, y_pred=prediction, average=average)
    precision = slmetrics.precision_score(y_true=dataset.test.cls, y_pred=prediction, average=average)
    f1_score = slmetrics.f1_score(y_true=dataset.test.cls, y_pred=prediction, average=average)
    if print_it:
        print("Metrics: Confusion Matrix, Precision, Recall, F1 score")
        print(confmat)
        print("metrics with average=", average)
        print('Precision:\t%.4f' % precision)
        print('Recall:\t\t%.4f' % recall)
        print('F1 score:\t%.4f' % f1_score)
    return confmat