import time
import datetime
import tensorflow.keras.backend as K
import os
from tqdm import tqdm
import tensorflow as tf
from model import (
    TagEncoder,
    TagDecoder,
    HarmonicCNN,
    WaveProjector,
    SupervisedClassifier,
)
from model import resemul

# fix random seed
SEED = 42
tf.random.set_seed(SEED)
from tensorflow.keras.metrics import Precision, Recall


def pairwise_loss(y, preds, margin=0.4):
    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss


class Solver(object):
    def __init__(self, config):
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        self.data_path = config.data_path
        self.withJE = config.withJE
        self.block = config.block
        self.dataset = config.dataset
        if self.dataset == "mtat":
            self.latent = 128
        elif self.dataset == "dcase":
            self.latent = 50
        elif self.dataset == "keyword":
            self.latent = 64

        self.input_length = 0
        self.batch_size = config.batch_size
        self.stage1_test_template = "AELoss : {:.5f}"
        self.stage1_train_template = (
            "Epoch: {}, TotalLoss : {:.4f},  AELoss : {:.4f}, PWLoss : {:.4f}"
        )
        self.stage2_test_template = "Test Loss : {}, Test AUC or F1 or ACC: {:.4f}"
        self.stage2_train_template = "Epoch : {}, Loss : {:.4f}, AUC or ACC: {:.4f}"
        self.encoder_type = config.encoder_type
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_save_path = config.model_save_path
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        model_save_dir_path = (
            self.model_save_path
            + self.dataset
            + "/"
            + self.encoder_type
            + "_"
            + self.block
            + "_"
            + str(self.withJE)
            + "/"
        )
        self.encoder_save_path = model_save_dir_path + "encoder"
        self.classifier_save_path = model_save_dir_path + "classifier"
        self.tag_encoder_save_path = model_save_dir_path + "tag_encoder"
        self.tag_decoder_save_path = model_save_dir_path + "tag_decoder"
        self.projecter_save_path = model_save_dir_path + "projector"

        # withJE가 True일 경우엔 tag encoder를 포함한 다른 모듈도 저장
        if self.withJE:
            if not os.path.exists(self.tag_encoder_save_path):
                os.makedirs(self.tag_encoder_save_path)
            if not os.path.exists(self.tag_decoder_save_path):
                os.makedirs(self.tag_decoder_save_path)
            if not os.path.exists(self.projecter_save_path):
                os.makedirs(self.projecter_save_path)

        if not os.path.exists(self.encoder_save_path):
            os.makedirs(self.encoder_save_path)

        if not os.path.exists(self.classifier_save_path):
            os.makedirs(self.classifier_save_path)

        self.train_ds, self.valid_ds, self.test_ds = self.get_data(self.data_path)
        self.isTest = config.isTest
        if self.isTest:
            self.stage1_epochs = [1, 1, 1]
            self.stage2_epochs = [1, 1, 1, 1]
        else:
            self.stage1_epochs = [60, 20, 20]
            self.stage2_epochs = [60, 20, 20, 100]

        self.get_model()
        self.get_optimizer()
        self.get_loss()
        self.get_metric()
        self.start_time = time.time()

    def get_data(self, root="../dataset"):
        if self.dataset == "mtat":
            from data_loader.mtat.loader import TrainLoader
            from data_loader.mtat.loader import TestLoader
        elif self.dataset == "dcase":
            from data_loader.dcase.loader import TrainLoader
            from data_loader.dcase.loader import TestLoader
        else:
            from data_loader.keyword.loader import TrainLoader
            from data_loader.keyword.loader import TestLoader

        if self.encoder_type == "HC":
            self.input_length = 80000
        else:
            self.input_length = 59049

        train_data = TrainLoader(
            root=root,
            tr_val="train",
            batch_size=self.batch_size,
            input_length=self.input_length,
        )
        valid_data = TestLoader(
            root=root,
            tr_val="valid",
            batch_size=self.batch_size,
            input_length=self.input_length,
        )
        test_data = TestLoader(
            root=root,
            tr_val="test",
            batch_size=self.batch_size,
            input_length=self.input_length,
        )
        return train_data, valid_data, test_data

    def get_model(self):
        if self.encoder_type == "HC":
            self.wave_encoder = HarmonicCNN()
        elif self.encoder_type == "SC":
            self.wave_encoder = resemul(block_type=self.block)
        elif self.encoder_type == "MS":
            self.wave_encoder = resemul(ms=True, block_type="rese")

        self.wave_projector = WaveProjector(self.latent)
        self.classifier = SupervisedClassifier(self.dataset)
        self.tag_encoder = TagEncoder(self.latent)
        self.tag_decoder = TagDecoder(self.latent, self.dataset)

    def get_optimizer(self):
        self.adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.sgd = tf.keras.optimizers.SGD(
            learning_rate=0.001, momentum=0.9, nesterov=True
        )
        self.sgd2 = tf.keras.optimizers.SGD(
            learning_rate=0.001, momentum=0.9, nesterov=True
        )

    def get_loss(self):
        if self.dataset == "mtat" or self.dataset == "dcase":
            self.c_loss = tf.keras.losses.BinaryCrossentropy()
        elif self.dataset == "keyword":
            self.c_loss = tf.keras.losses.CategoricalCrossentropy()
        self.ae_loss = tf.keras.losses.MeanSquaredError()
        self.pw_loss = tf.keras.losses.CosineSimilarity()

    def get_metric(self):
        self.s1_train_loss = tf.keras.metrics.Mean()
        self.s1_train_ae = tf.keras.metrics.Mean()
        self.s1_train_pw = tf.keras.metrics.Mean()
        self.s1_test_loss = tf.keras.metrics.Mean()
        self.s2_train_loss = tf.keras.metrics.Mean()
        self.s2_test_loss = tf.keras.metrics.Mean()
        if self.dataset == "mtat":
            self.s2_train_auc = tf.keras.metrics.AUC()
            self.s2_test_auc = tf.keras.metrics.AUC()
            self.s2_train_pr = tf.keras.metrics.AUC(curve="PR")
            self.s2_test_pr = tf.keras.metrics.AUC(curve="PR")

        elif self.dataset == "keyword":
            self.s2_train_auc = tf.keras.metrics.CategoricalAccuracy()
            self.s2_test_auc = tf.keras.metrics.CategoricalAccuracy()
            self.s2_train_pr = tf.keras.metrics.CategoricalAccuracy()
            self.s2_test_pr = tf.keras.metrics.CategoricalAccuracy()

        elif self.dataset == "dcase":
            self.s2_train_auc = tf.keras.metrics.AUC()
            self.s2_test_auc = tf.keras.metrics.AUC()
            self.s2_train_pr = tf.keras.metrics.AUC(curve="PR")
            self.s2_test_pr = tf.keras.metrics.AUC(curve="PR")

    @tf.function
    def stage1_adam_step(self, wave, labels):
        # apply GradientTape for differentiation
        with tf.GradientTape() as tape:
            # 1. prediction
            # wave
            # z1 shape : (1024,)
            # r1 shape : (128,)
            z1 = self.wave_encoder(wave, training=True)
            r1 = self.wave_projector(z1, training=True)
            # tag autoencoder
            # z2 shape : (128,)
            # predictions = (50,)
            z2 = self.tag_encoder(labels, training=True)
            predictions = self.tag_decoder(z2, training=True)

            # 2. calculate Loss
            # ae_loss : autoencoder loss
            # pw_loss : loss between rese and autoencoder
            ae_loss = self.ae_loss(labels, predictions)
            # pw_loss = (1-self.pw_loss(z2, r1)) + (1-self.pw_loss(1-z2, 1-r1))
            pw_loss = pairwise_loss(z2, r1)
            total_loss = ae_loss + pw_loss

        train_variable = (
            self.wave_encoder.trainable_variables
            + self.wave_projector.trainable_variables
            + self.tag_encoder.trainable_variables
            + self.tag_decoder.trainable_variables
        )

        # 3. calculate gradients
        gradients = tape.gradient(total_loss, train_variable)

        # 4. Backpropagation - update weight
        self.adam.apply_gradients(zip(gradients, train_variable))

        # update loss and accuracy
        self.s1_train_ae(ae_loss)
        self.s1_train_pw(pw_loss)
        self.s1_train_loss(total_loss)

    @tf.function
    def stage1_sgd_step(self, wave, labels):
        # apply GradientTape for differentiation
        with tf.GradientTape() as tape:
            # 1. prediction
            # wave
            # z1 shape : (1024,)
            # r1 shape : (128,)
            z1 = self.wave_encoder(wave, training=True)
            r1 = self.wave_projector(z1, training=True)
            # tag autoencoder
            # z2 shape : (128,)
            # predictions = (50,)
            z2 = self.tag_encoder(labels, training=True)
            predictions = self.tag_decoder(z2, training=True)

            # 2. calculate Loss
            # ae_loss : autoencoder loss
            # pw_loss : loss between rese and autoencoder
            ae_loss = self.ae_loss(labels, predictions)
            # pw_loss = (1-self.pw_loss(z2, r1)) + (1-self.pw_loss(1-z2, 1-r1))
            pw_loss = pairwise_loss(z2, r1)
            total_loss = ae_loss + pw_loss

        train_variable = (
            self.wave_encoder.trainable_variables
            + self.wave_projector.trainable_variables
            + self.tag_encoder.trainable_variables
            + self.tag_decoder.trainable_variables
        )

        # 3. calculate gradients
        gradients = tape.gradient(total_loss, train_variable)

        # 4. Backpropagation - update weight
        self.sgd.apply_gradients(zip(gradients, train_variable))

        # update loss and accuracy
        self.s1_train_ae(ae_loss)
        self.s1_train_pw(pw_loss)
        self.s1_train_loss(total_loss)

    def stage1_train(self, epochs, optimizer):
        for epoch in range(epochs):
            for wave, labels in tqdm(self.train_ds):
                if optimizer == "adam":
                    self.stage1_adam_step(wave, labels)
                elif optimizer == "sgd":
                    self.stage1_sgd_step(wave, labels)

            stage1_log = self.stage1_train_template.format(
                epoch + 1,
                self.s1_train_ae.result(),
                self.s1_train_pw.result(),
                self.s1_train_loss.result(),
            )
            print(stage1_log)

    @tf.function
    def stage1_test_step(self, wave, labels):
        z = self.tag_encoder(labels, training=False)
        predictions = self.tag_decoder(z, training=False)
        recon_loss = self.ae_loss(labels, predictions)
        self.s1_test_loss(recon_loss)

    @tf.function
    def stage2_adam_step(self, wave, labels):
        with tf.GradientTape() as tape:

            z = self.wave_encoder(wave, training=False)
            predictions = self.classifier(z, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = self.classifier.trainable_variables
        gradients = tape.gradient(loss, train_variable)

        self.adam.apply_gradients(zip(gradients, train_variable))

        self.s2_train_loss(loss)
        self.s2_train_auc(labels, predictions)
        self.s2_train_pr(labels, predictions)

    @tf.function
    def stage2_sgd_step(self, wave, labels):
        with tf.GradientTape() as tape:

            z = self.wave_encoder(wave, training=False)
            predictions = self.classifier(z, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = self.classifier.trainable_variables
        gradients = tape.gradient(loss, train_variable)

        self.sgd2.apply_gradients(zip(gradients, train_variable))

        self.s2_train_loss(loss)
        self.s2_train_auc(labels, predictions)
        self.s2_train_pr(labels, predictions)

    @tf.function
    def noJE_adam_step(self, wave, labels):
        with tf.GradientTape() as tape:
            z = self.wave_encoder(wave, training=True)

            predictions = self.classifier(z, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = (
            self.wave_encoder.trainable_variables + self.classifier.trainable_variables
        )
        gradients = tape.gradient(loss, train_variable)

        self.adam.apply_gradients(zip(gradients, train_variable))

        self.s2_train_loss(loss)
        self.s2_train_auc(labels, predictions)
        self.s2_train_pr(labels, predictions)

    @tf.function
    def noJE_sgd_step(self, wave, labels):
        with tf.GradientTape() as tape:
            z = self.wave_encoder(wave, training=True)
            predictions = self.classifier(z, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = (
            self.wave_encoder.trainable_variables + self.classifier.trainable_variables
        )
        gradients = tape.gradient(loss, train_variable)

        self.sgd2.apply_gradients(zip(gradients, train_variable))

        self.s2_train_loss(loss)
        self.s2_train_auc(labels, predictions)
        self.s2_train_pr(labels, predictions)

    @tf.function
    def stage2_test_step(self, wave, labels):
        z = self.wave_encoder(wave, training=False)
        predictions = self.classifier(z, training=False)

        loss = self.c_loss(labels, predictions)
        self.s2_test_loss(loss)
        self.s2_test_auc(labels, predictions)
        self.s2_test_pr(labels, predictions)

    def stage2_train(self, epochs, optimizer, withJE):
        for epoch in range(epochs):
            for wave, labels in tqdm(self.train_ds):
                if optimizer == "adam":
                    if withJE:
                        self.stage2_adam_step(wave, labels)
                    else:
                        self.noJE_adam_step(wave, labels)
                elif optimizer == "sgd":
                    if withJE:
                        self.stage2_sgd_step(wave, labels)
                    else:
                        self.noJE_sgd_step(wave, labels)

            stage2_log = self.stage2_train_template.format(
                epoch + 1,
                self.s2_train_loss.result(),
                self.s2_train_auc.result(),
            )
            print(stage2_log)

    def run_stage1(self):
        print("@@@@@@@@@@@@@@@@@@@Start training Stage 1@@@@@@@@@@@@@@@@@@\n")
        # training
        for i in range(3):
            if i == 0:
                epochs = self.stage1_epochs[0]
                self.stage1_train(epochs, optimizer="adam")
            elif i == 1:
                epochs = self.stage1_epochs[1]
                self.stage1_train(epochs, optimizer="sgd")
            else:
                epochs = self.stage1_epochs[2]
                new_lr = 0.0001
                self.sgd.lr.assign(new_lr)
                self.stage1_train(epochs, optimizer="sgd")
        for wave, labels in tqdm(self.test_ds):
            self.stage1_test_step(wave, labels)

        stage1_test_loss = self.stage1_test_template.format(self.s1_test_loss.result())
        print(stage1_test_loss)

    def run_stage2(self, withJE):
        print("\n\n@@@@@@@@@@@@@@@@@@@Start training Stage 2@@@@@@@@@@@@@@@@@@\n")
        for i in range(4):
            if i == 0:
                epochs = self.stage2_epochs[0]
                self.stage2_train(epochs=epochs, optimizer="adam", withJE=withJE)
            elif i == 1:
                epochs = self.stage2_epochs[1]
                self.stage2_train(epochs=epochs, optimizer="sgd", withJE=withJE)
            elif i == 2:
                epochs = self.stage2_epochs[2]
                new_lr = 0.0001
                self.sgd2.lr.assign(new_lr)
                self.stage2_train(epochs=epochs, optimizer="sgd", withJE=withJE)
            else:
                epochs = self.stage2_epochs[3]
                new_lr = 0.00001
                self.sgd2.lr.assign(new_lr)
                self.stage2_train(epochs=epochs, optimizer="sgd", withJE=withJE)

        for wave, labels in tqdm(self.test_ds):
            self.stage2_test_step(wave, labels)

        print("Time taken : ", time.time() - self.start_time)

        f1 = (
            2
            * self.s2_test_auc.result()
            * self.s2_test_pr.result()
            / (self.s2_test_auc.result() + self.s2_test_pr.result())
        )
        print("[Result] Test loss: ", self.s2_test_loss.result())
        print("[Result] Test roc-auc or acc: ", self.s2_test_auc.result())
        print("[Result] Test pr-auc or acc: ", self.s2_test_pr.result())
        print("[Result] Test F1 or acc: ", f1)

    def run(self):
        print(self.withJE)
        if self.withJE:
            self.run_stage1()
            self.tag_encoder.save(self.tag_encoder_save_path)
            self.tag_decoder.save(self.tag_decoder_save_path)
            self.wave_projector.save(self.projecter_save_path)
        self.run_stage2(withJE=self.withJE)
        self.wave_encoder.save(self.encoder_save_path)
        self.classifier.save(self.classifier_save_path)
