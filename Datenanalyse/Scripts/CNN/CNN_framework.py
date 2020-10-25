import tensorflow as tf
import numpy as np
import sklearn.metrics as slmetrics
import time, sys, os
from datetime import timedelta
import utilities.plot as plotlib
from models import models_1D, models_2D
import math


def run(conf, dataset, classlabels):
    """main method to run the whole training and evaluation pipeline"""
    if conf.write_file or conf.plot:
        if not os.path.isdir(conf.logspath):
            os.makedirs(conf.logspath)  # make sure directory exists for writing the log file
    # toggle print
    if conf.write_file:
        orig_stdout = sys.stdout    # store original stdout
        f = open(conf.logspath + 'log.txt', 'w+')
        sys.stdout = f
    # print the used parameter setup
    for arg in vars(conf):
        print(arg, getattr(conf, arg))

    def loss(out, y):
        """calculate loss on given net output (compared with true labels y)"""
        with tf.name_scope("Loss"):
            # using exclusive `labels` (wherein one and only one class is true at a time)
            labels = tf.argmax(y, axis=1)    # encode classes of labels in one vector of length=batch_size
            xentropy = tf.losses.sparse_softmax_cross_entropy(labels, out, scope="CrossEntropy")
            # xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y, name="CrossEntropy")
            loss_l2 = tf.losses.get_regularization_loss()    # https://stackoverflow.com/questions/44232566/add-l2-regularization-when-using-high-level-tf-layers
            total_loss = tf.reduce_mean(xentropy) + loss_l2
        tf.summary.scalar("L2_regularization_sum", loss_l2)
        tf.summary.scalar("cross_entropy", xentropy)
        return total_loss

    def training_step(costs, step):
        """do one optimization step (minimize costs)"""
        tf.summary.scalar("Cost", costs)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_operation = optimizer.minimize(costs, global_step=step)
        return train_operation

    def evaluate(out, y_tr):
        """calculate accuracy"""
        with tf.name_scope("Accuracy"):
            correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_tr, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("Error", (1.0 - accuracy))
            tf.summary.scalar("Accuracy", accuracy)
            return accuracy

    def feed_dict(num, batchsize=conf.batch_size):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders. (1=train, 2=val, 3=test)"""
        if num == 1:
            # Get a batch of training examples.
            xs, ys = dataset.train.next_batch(batchsize)    # includes shuffling
            k = conf.keep_prob
            tr = True
        elif num == 2:
            xs, ys = dataset.validation.data, dataset.validation.labels
            k = 1.0     # do not use dropout for validation
            tr = False
        else:
            xs, ys = dataset.test.next_batch(batchsize, shuffle=False)  # disable shuffle
            k = 1.0     # do not use dropout for test
            tr = False
        # Put the data into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        return {x: xs, y_true: ys, keep_prob: k, is_training_phase: tr}

    with tf.variable_scope("CNN_Model"):
        with tf.name_scope("Inputs"):
            x = tf.placeholder(tf.float32, shape=[None, conf.seq_lngth * conf.num_ch], name='x')
            y_true = tf.placeholder(tf.float32, shape=[None, conf.num_cl], name='y_true')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout probability
            is_training_phase = tf.placeholder(tf.bool, name='is_training_phase')
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if conf.enable_lr_decay:
            learning_rate = tf.train.exponential_decay(learning_rate=conf.lr,
                                                       global_step=global_step,
                                                       decay_steps=2500,
                                                       decay_rate=0.1,
                                                       staircase=False)
            # learning_rate = tf.train.polynomial_decay(conf.lr, global_step, 3000, end_learning_rate=0.00001)
            tf.summary.scalar("learning_rate", learning_rate)
        else:
            learning_rate = conf.lr
        # initialization of some variables
        val_err_es = 10000.  # high value for first comparison for early stopping
        val_cost_es = 10000.
        traincost = 0.
        patience_cnt = 0

        # computational graph
        if conf.input == '2D':
            output = models_2D.conv_net(x, keep_prob, is_training_phase, conf)
        else:
            output = models_1D.conv_net(x, keep_prob, is_training_phase, conf)  # use placeholders, rest is static (conf)
        cost = loss(output, y_true)
        # control_dependencies with update required for proper Batch Normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = training_step(cost, global_step)
        eval_op = evaluate(output, y_true)

        # save + enable tensorboard
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)     # log variables
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=1)    # default: save latest five checkpoints
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True    # avoid CUDNN_STATUS_INTERNAL_ERROR (do not allocate too much memory at once)
        config.allow_soft_placement = True  # use cpu if gpu does not exist
        session = tf.Session(config=config)  # instantiate session
        summary_writer_train = tf.summary.FileWriter(conf.logspath + 'graph/train', session.graph)
        summary_writer_val = tf.summary.FileWriter(conf.logspath + 'graph/val')    # omit graph here

        # initialize Variables
        init_op = tf.global_variables_initializer()
        session.run(init_op)
        tf.train.export_meta_graph(conf.logspath + 'metagraph.meta')

        # Start-time used for printing time-usage below.
        start_time = time.time()

        # Training cycle
        with tf.device('/gpu:0'):
            for epoch in range(conf.num_epochs):
                avg_cost, avg_trainacc = 0., 0.

                total_batch = int(dataset.train.num_examples / conf.batch_size)
                for i in range(total_batch):    # loop over all mini-batches
                    if i % conf.disp_train == (conf.disp_train - 1):   # record execution stats every n steps
                        if conf.metadata:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                        # train and calculate accuracy on training set
                        summary_str, _, trainacc, batch_cost = session.run([summary_op, train_op, eval_op, cost],
                                                                           feed_dict=feed_dict(1),
                                                                           options=run_options if conf.metadata else None,
                                                                           run_metadata=run_metadata if conf.metadata else None)
                        if conf.metadata:
                            summary_writer_train.add_run_metadata(run_metadata, np.unicode(session.run(global_step)))
                        summary_writer_train.add_summary(summary_str, session.run(global_step))
                        # Message for printing.
                        msg = "Optimization Iteration (minibatch-#): {0:>6}, Training Accuracy: {1:>6.1%}"
                        # Print it.
                        print(msg.format(i, trainacc))
                    else:   # otherwise just do the training step
                        _, batch_cost, trainacc = session.run([train_op, cost, eval_op],
                                                              feed_dict=feed_dict(1))
                    # Compute average cost
                    avg_cost += batch_cost/total_batch
                    avg_trainacc += trainacc/total_batch
                # Print per epoch
                if epoch % 1 == 0:
                    print("Epoch:", '%04d' % epoch, "avg. train acc. =", "{:.5f}".format(avg_trainacc))
                    print("Epoch:", '%04d' % epoch, "training cost =", "{:.9f}".format(avg_cost))
                    # Message for printing.
                    # Calculate accuracy on validation set
                    summary_str, val_acc, val_cost = session.run([summary_op, eval_op, cost], feed_dict=feed_dict(2))
                    # write summaries
                    summary_writer_val.add_summary(summary_str, session.run(global_step))
                    # Print it.
                    val_err = 1 - val_acc
                    print("Epoch:", '%04d' % epoch, "validation cost =", "{:.9f}".format(val_cost))
                    print("Validation Error:", val_err)

                    # Early Stopping
                    # if val_cost_es - val_cost > conf.es_mindelta:    # check for early stopping criterion
                    if val_err_es - val_err > conf.es_mindelta:
                        patience_cnt = 0    # reset counter for patience
                        saved_epoch = epoch   # store epoch for best validation error
                        saved_step = session.run(global_step)
                        val_cost_es = val_cost    # save currently best validation error
                        val_err_es = val_err
                        trainacc_es = avg_trainacc
                        # save latest good model
                        saver.save(session, conf.logspath + "Model-checkpoint", global_step=global_step, write_meta_graph=False)
                    else:
                        patience_cnt += 1   # if no great ( < delta) improvement in validation cost --> increase counter
                    if patience_cnt > conf.es_patience:  # check for early stopping patience
                        print("Early Stopping in Epoch %04d, resetting to model from epoch %04d (step %05d)" % (epoch, saved_epoch, saved_step))
                        msg = "Saved Validation Error: {0:>6.4}, last Validation Error: {1:>6.4}"
                        msg2 = "Saved Validation Cost: {0:>6.4}, last Validation Cost: {1:>6.4}"
                        print(msg.format(val_err_es, val_err))
                        print(msg2.format(val_cost_es, val_cost))
                        saver.restore(session, tf.train.latest_checkpoint(conf.logspath))
                        break
            # Print time for training and number of parameters
            print("Optimization Finished!")
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
            print("Number of trainable parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))



        ######## evaluate validation data
        valacc, val_predictions, scores_val = session.run(
            [eval_op, tf.argmax(tf.nn.softmax(output), axis=1), tf.nn.softmax(output)],
            feed_dict=feed_dict(2))  # use full mini-batch size
        dataset.validation.cls = np.argmax(dataset.validation.labels, axis=1)
        if conf.write_file:
            # save prediction scores and labels for final test procedure
            np.savetxt(conf.logspath + 'PredictionScores_val.csv', scores_val, delimiter=",")
            np.savetxt(conf.logspath + 'PredictionLabels_val.csv', val_predictions, fmt='%d')
            np.savetxt(conf.logspath + 'TrueLabels_val.csv', dataset.validation.cls, delimiter=",")
            np.savetxt(conf.logspath + 'Index_val.csv', dataset.validation.index, delimiter=",")
        confmat = evaluate_metrics_val(dataset, val_predictions, classlabels, conf.metrics_avg)
        print("Accuracy:\t%.4f" % valacc)
        # if conf.evaluate and conf.write_file:

        if conf.plot:
            plotlib.plot_confusionmatrix(confmat, conf.logspath, classlabels, export=conf.export, filenamePostfix='_validation')



        ######## evaluate test data
        testacc = 0.
        test_predictions = []
        total_batch_test = int(math.ceil(dataset.test.num_examples / conf.batch_size))  # make sure to include fractional mini-batch
        remainder = dataset.test.num_examples % conf.batch_size     # how many samples are in the incomplete mini-batch?
        for i in range(total_batch_test):  # loop over all mini-batches
            if i == total_batch_test-1 and remainder != 0:  # in last of all mini-batches and if there is a remainder
                batch_testacc, test_pred, score = session.run([eval_op, tf.argmax(tf.nn.softmax(output), axis=1), tf.nn.softmax(output)],
                                                       feed_dict=feed_dict(3, remainder))   # use reduced mini-batch size
            else:
                batch_testacc, test_pred, score = session.run([eval_op, tf.argmax(tf.nn.softmax(output), axis=1), tf.nn.softmax(output)],
                                                       feed_dict=feed_dict(3))  # use full mini-batch size
            testacc += batch_testacc/total_batch_test   # calculate test accuracy over all the mini-batches
            test_predictions.extend(test_pred)  # merge predictions into one list
            if i == 0:
                scores = score
            else:
                scores = np.append(scores, score, axis=0)
        confmat = evaluate_metrics_test(dataset, test_predictions, classlabels, conf.metrics_avg)
        print("Accuracy:\t%.4f" % testacc)
        # if conf.evaluate and conf.write_file:
        if conf.write_file:
            # save prediction scores and labels for final test procedure
            np.savetxt(conf.logspath + 'PredictionScores.csv', scores, delimiter=",")
            np.savetxt(conf.logspath + 'PredictionLabels.csv', test_predictions, fmt='%d')
            np.savetxt(conf.logspath + 'TrueLabels.csv', dataset.test.cls, delimiter=",")
            np.savetxt(conf.logspath + 'Index.csv', dataset.test.index, delimiter=",")
    if conf.plot:
        plotlib.plot_confusionmatrix(confmat, conf.logspath, classlabels, export=conf.export)
        try:
            plotlib.plot_tensorboard_scalar(conf.logspath, 'Error', export=conf.export)    # plot error curve
            plotlib.plot_tensorboard_scalar(conf.logspath, 'Cost', export=conf.export)    # plot cost curve
        except KeyError:
            print('Could not find `Error´ or `Cost´ in tensorboard log file. Training did not converge in a single iteration.')
        try:
            # plot random sample from train dataset and conv layer for that sample
            plotlib.plot_conv_layer_1d(session, x, dataset.train, 2, conf.logspath, layer=conf.featureviz)
            plotlib.plot_conv_layer_1d_spectrum(session, x, dataset.train, 2, conf.logspath, layer=conf.featureviz)
            layername = 'conv1_relu_bn'
            plotlib.plot_conv_layer_1d_weights(session, layername, conf.logspath)
        except:
            print('Could not plot random samples and layer internals')
    # finish session
    summary_writer_train.close()
    summary_writer_val.close()
    tf.reset_default_graph()
    session.close()

    if conf.write_file:
        sys.stdout = orig_stdout    # restore stdout
        f.close()   # close file
    return testacc, trainacc_es, avg_trainacc, time_dif, 1-val_err_es


def evaluate_metrics_test(dataset, prediction, labels, average, print_it=True):
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
        print('Sequence of class labels: ', labels)
        print(confmat)
        print("metrics with average=", average)
        print('Precision:\t%.4f' % precision)
        print('Recall:\t\t%.4f' % recall)
        print('F1 score:\t%.4f' % f1_score)
    return confmat

def evaluate_metrics_val(dataset, prediction, labels, average, print_it=True):
    """calculate confusion matrix, recall, precision, f1 score"""
    # get classes instead of one-hot encoding
    dataset.validation.cls = np.argmax(dataset.validation.labels, axis=1)
    # calculate metrics
    confmat = slmetrics.confusion_matrix(y_true=dataset.validation.cls, y_pred=prediction)
    recall = slmetrics.recall_score(y_true=dataset.validation.cls, y_pred=prediction, average=average)
    precision = slmetrics.precision_score(y_true=dataset.validation.cls, y_pred=prediction, average=average)
    f1_score = slmetrics.f1_score(y_true=dataset.validation.cls, y_pred=prediction, average=average)
    if print_it:
        print("Metrics: Confusion Matrix, Precision, Recall, F1 score")
        print('Sequence of class labels: ', labels)
        print(confmat)
        print("metrics with average=", average)
        print('Precision:\t%.4f' % precision)
        print('Recall:\t\t%.4f' % recall)
        print('F1 score:\t%.4f' % f1_score)
    return confmat