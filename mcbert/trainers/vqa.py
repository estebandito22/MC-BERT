"""Classes to train Deep Source Separation Models."""

import os
import csv
import random

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.optim import Adam

from pytorch_pretrained_bert import BertAdam

from mcbert.trainers.base_trainer import Trainer
from mcbert.models.mcbert import MCBertModel
from mcbert.models.classifier_head import ClassifierHeadModel
from mcbert.models.mcboriginal import MCBOriginalModel
from mcbert.models.layers.embedding.glove_embedder import GloveEmbedder
from mcbert.models.layers.embedding.elmo_embedder import ElmoEmbedder


class VQATrainer(Trainer):

    """Class to train and evaluate BertVisualMemory."""

    def __init__(self, model_type='mc-bert', vis_feat_dim=2048, spatial_size=7,
                 lm_hidden_dim=768, cmb_feat_dim=16000, kernel_size=3,
                 dropout=0.2, n_classes=3000, batch_size=64,
                 learning_rate=3e-5, warmup_proportion=0.1, num_epochs=100, vocab=None,
                 use_attention=True, use_external_MCB=True, use_batchnorm=False,
                 weight_decay=1e-6, lm_only=False, use_MCB_init=False,
                 normalize_vis_feats=False, patience=10, min_lr=0, freeze_epoch=None):
        """
        Initialize BertMBC model.

        Args
        ----
            model_type : string, model name, 'mc-bert'.
            vis_feat_dim : int, intermediate visual feature dimension.
            spatial_size : int, spatial size of visual features.
            lm_hidden_dim : int, size of hidden state in language model.
            cmb_feat_dim : int, combined feature dimension.
            kernel_size : int, kernel_size to use in attention.
            batch_size : int, batch size for optimization.
            num_epochs : int, number of epochs to train for.

        """
        # Trainer attributes
        self.model_type = model_type
        self.vis_feat_dim = vis_feat_dim
        self.spatial_size = spatial_size
        self.lm_hidden_dim = lm_hidden_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.num_epochs = num_epochs
        self.use_attention = use_attention
        self.use_external_MCB = use_external_MCB
        self.use_batchnorm = use_batchnorm
        self.weight_decay = weight_decay
        self.lm_only = lm_only
        self.use_MCB_init = use_MCB_init
        self.normalize_vis_feats = normalize_vis_feats
        self.patience = patience
        self.min_lr = min_lr
        self.freeze_epoch = freeze_epoch

        # Model attributes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.nn_epoch = 0
        self.best_val_acc = 0
        self.save_dir = None
        self.vocab = vocab

        # reproducability attributes
        self.torch_rng_state = None
        self.numpy_rng_state = None

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self, train_chunks, train_dataset_len=None):
        """Initialize the nn model for training."""
        if self.model_type == 'mc-bert':
            mcb_model = MCBertModel(
                vis_feat_dim=self.vis_feat_dim, spatial_size=self.spatial_size,
                hidden_dim=self.lm_hidden_dim, cmb_feat_dim=self.cmb_feat_dim,
                kernel_size=self.kernel_size, classification=True,
                use_attention=self.use_attention, use_external_MCB=self.use_external_MCB,
                use_batchnorm=self.use_batchnorm, lm_only=self.lm_only,
                normalize_vis_feats=self.normalize_vis_feats)
        elif self.model_type == 'mcb' or self.model_type == 'mcb-bi':
            bidi = True if self.model_type == 'mcb-bi' else False
            embedder = GloveEmbedder(self.vocab, 300)
            mcb_model = MCBOriginalModel(embedder,
                vis_feat_dim=self.vis_feat_dim, spatial_size=self.spatial_size,
                hidden_dim=self.lm_hidden_dim, cmb_feat_dim=self.cmb_feat_dim,
                kernel_size=self.kernel_size, bidirectional=bidi,classification=True,
                use_attention=self.use_attention, use_external_MCB=self.use_external_MCB,
                use_batchnorm=self.use_batchnorm, lm_only=self.lm_only,
                use_MCB_init=self.use_MCB_init, normalize_vis_feats=self.normalize_vis_feats)
        elif self.model_type == 'mc-elmo':
            embedder = ElmoEmbedder()
            mcb_model = MCBOriginalModel(embedder,
                 vis_feat_dim=self.vis_feat_dim, spatial_size=self.spatial_size,
                 hidden_dim=self.lm_hidden_dim, cmb_feat_dim=self.cmb_feat_dim,
                 kernel_size=self.kernel_size, bidirectional=True, classification=True,
                 use_attention=self.use_attention, use_external_MCB=self.use_external_MCB,
                 use_batchnorm=self.use_batchnorm, lm_only=self.lm_only,
                 use_MCB_init=self.use_MCB_init, normalize_vis_feats=self.normalize_vis_feats)

        else:
            raise ValueError("Did not recognize model type!")

        self.model = ClassifierHeadModel(
            mcb_model, dropout=self.dropout, n_classes=self.n_classes)

        if self.model_type == 'mc-bert':
            # Prepare optimizer
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            our_decay = ['bn1.weight', 'bn2.weight', 'conv1.weight',
                         'conv2.weight', 'cls.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay+our_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in our_decay)],
                 'weight_decay': self.weight_decay}
                ]

            self.optimizer = BertAdam(
                optimizer_grouped_parameters, lr=self.learning_rate,
                warmup=self.warmup_proportion,
                t_total=int(train_dataset_len / train_chunks
                            / self.batch_size * self.num_epochs))
            self.scheduler = None
        else:
            self.optimizer = Adam(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'max', verbose=True, patience=self.patience,
                min_lr=self.min_lr)

        if self.USE_CUDA:
            self.model = self.model.cuda()

        # reproducability and deteriministic continuation of models
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.torch_rng_state = torch.get_rng_state()
        self.numpy_rng_state = np.random.get_state()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        correct = 0
        loss_fct = torch.nn.NLLLoss(ignore_index=int(self.n_classes - 1))

        for batch_samples in tqdm(loader):

            # prepare training sample
            # batch_size x seqlen
            input_ids = batch_samples['input_ids']
            token_type_ids = batch_samples['token_type_ids']
            attention_mask = batch_samples['attention_mask']
            # batch_size
            labels = batch_samples['labels']

            if self.nn_epoch >= self.freeze_epoch:
                lm_feats = batch_samples['lm_feats']
            else:
                lm_feats = None

            # batch_size x seqlen x channel x height x width
            vis_feats = batch_samples['vis_feats']

            if self.USE_CUDA:
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                vis_feats = vis_feats.cuda()

            # forward pass
            self.model.zero_grad()
            #let's calculate loss and accuracy out here
            lm_feats, logits = self.model(
                vis_feats, input_ids, token_type_ids, attention_mask, None, lm_feats)

            if self.model_type == 'mcbert' and self.nn_epoch >= (self.freeze_epoch - self.train_chunks) and self.nn_epoch < self.freeze_epoch:
                self.train_dataset.save_sentence_tensor(input_ids, lm_feats.detach(), os.path.join(self.save_dir, self.model_dir))

            probs = torch.nn.functional.log_softmax(logits, dim=1)
            loss = loss_fct(probs, labels)

            # backward pass
            loss.backward()
            self.optimizer.step()

            if random.random() > 1:
                print("\nconv1 grad:", torch.norm(self.model.mcb_model.attention.conv1.weight.grad, p=2))
                print("conv2 grad:", torch.norm(self.model.mcb_model.attention.conv2.weight.grad, p=2))
                print("lstm0i grad:", torch.norm(self.model.mcb_model.lstm.weight_ih_l0.grad, p=2))
                print("lstm0h grad:", torch.norm(self.model.mcb_model.lstm.weight_hh_l0.grad, p=2))
                print("lstm1 grad:", torch.norm(self.model.mcb_model.lstm.weight_ih_l1.grad, p=2))
                print("cls grad:", torch.norm(self.model.cls.weight.grad, p=2))
            # compute train loss and acc
            predicts = torch.argmax(probs, dim=1)
            correct += torch.sum(predicts == labels).item()
            bs = input_ids.size(0)
            samples_processed += bs
            train_loss += loss.item() * bs

        train_loss /= samples_processed
        acc = correct / samples_processed

        return train_loss, acc

    def _eval_epoch(self, loader, outfile=None):
        """Eval epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        correct = 0
        loss_fct = torch.nn.NLLLoss()

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare training sample
                # batch_size x seqlen
                input_ids = batch_samples['input_ids']
                token_type_ids = batch_samples['token_type_ids']
                attention_mask = batch_samples['attention_mask']
                # batch_size
                labels = batch_samples['labels']
                # batch_size x seqlen x channel x height x width
                vis_feats = batch_samples['vis_feats']

                if self.USE_CUDA:
                    input_ids = input_ids.cuda()
                    token_type_ids = token_type_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    labels = labels.cuda()
                    vis_feats = vis_feats.cuda()

                # forward pass
                # let's calculate loss and accuracy out here
                _, logits = self.model(
                    vis_feats, input_ids, token_type_ids, attention_mask, None)
                probs = torch.nn.functional.log_softmax(logits, dim=1)
                loss = loss_fct(probs, labels)

                # compute train loss and acc
                predicts = torch.argmax(probs, dim=1)
                correct += torch.sum(predicts == labels).item()
                #write out results
                if outfile is not None:
                    qids = batch_samples['qids']
                    for i in range(len(qids)):
                        outfile[0].writerow([qids[i].item(), predicts[i].item(), labels[i].item()])
                    outfile[1].flush()

                if random.random() > 1:
                    print("")
                    for i in range(len(predicts)):
                        print(predicts[i].item(), labels[i].item(), sep=':')

                bs = input_ids.size(0)
                samples_processed += bs
                val_loss += loss.item() * bs

            val_loss /= samples_processed
            acc = correct / samples_processed

        return val_loss, acc

    def fit(self, train_dataset, train_chunks, val_dataset, eval_pct, save_dir, warm_start=False):
        """
        Train the NN model.

        Args
        ----
            train_dataset : PyTorch dataset, training data.
            val_dataset : PyTorch dataset, validation data.
            save_dir: directory to save nn_model

        """
        # Print settings to output file
        print("Settings:\n\
               Model Type: {}\n\
               Language Model Only: {}\n\
               Visual Feature Dimension: {}\n\
               Spatial Size: {}\n\
               LM Hidden Dimension: {}\n\
               Combined Feature Dimension: {}\n\
               Kernel Size: {}\n\
               Dropout: {}\n\
               Weight Decay: {}\n\
               Learning Rate: {}\n\
               Patience: {}\n\
               Min LR: {}\n\
               Batch Size: {}\n\
               Chunks per epoch: {}\n\
               Eval Pct: {}\n\
               Warmup Proportion: {}\n\
               N Classes: {}\n\
               Use Attention: {}\n\
               Use External MCB: {}\n\
               Use Batchnorm: {}\n\
               Use MCBPaper Init: {}\n\
               Normalize Visual Features: {}\n\
               Freeze Epoch: {}\n\
               Save Dir: {}".format(
                   self.model_type, self.lm_only, self.vis_feat_dim, self.spatial_size,
                   self.lm_hidden_dim, self.cmb_feat_dim, self.kernel_size,
                   self.dropout, self.weight_decay, self.learning_rate,
                   self.patience, self.min_lr, self.batch_size, train_chunks, eval_pct,
                   self.warmup_proportion, self.n_classes, self.use_attention,
                   self.use_external_MCB, self.use_batchnorm,
                   self.use_MCB_init, self.normalize_vis_feats, self.freeze_epoch,
                   save_dir), flush=True)

        self.save_dir = save_dir
        self.model_dir = self._format_model_subdir()

        #need this to save before freezing
        self.train_chunks = train_chunks
        self.train_dataset = train_dataset

        # initialize constant loaders
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=8)

        # initialize neural network and training variables
        if not warm_start:
            self._init_nn(train_chunks, len(train_dataset))
        train_loss = 0
        train_acc = 0

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            train_loaders = self._batch_loaders(train_dataset, k=train_chunks)

            for train_loader in train_loaders:
                if self.nn_epoch > 0:
                    print("\nInitializing train epoch...", flush=True)
                    train_loss, train_acc = self._train_epoch(train_loader)

                print("\nInitializing val epoch...", flush=True)
                val_loss, val_acc = self._eval_epoch(val_loader)

                # report
                print("\nEpoch: [{}/{}]\tTrain Loss: {}\tTrain Acc: {}\tVal Loss: {}\tVal Acc: {}".format(
                    self.nn_epoch, self.num_epochs, np.round(train_loss, 5),np.round(train_acc * 100, 2),
                    np.round(val_loss, 5), np.round(val_acc * 100, 2)), flush=True)

                # save best
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.torch_rng_state = torch.get_rng_state()
                    self.numpy_rng_state = np.random.get_state()
                    self.save()
                self.nn_epoch += 1

                if self.scheduler:
                    self.scheduler.step(val_acc)

    def score(self, loader):
        """
        Score all predictions.

        Args
        ----
            loader : PyTorch DataLoader.

        """
        self.model.eval()
        raise NotImplementedError("Not yet implemented!")

    def predict(self, loader):
        """
        Predict for an input.

        Args
        ----
            loader : PyTorch DataLoader.

        """
        self.model.eval()
        raise NotImplementedError("Not yet implemented!")

    def report_results(self, val_dataset, outfile_name):

        # grabbing 10%, could be smarter about this...
        #val_dataset = Subset(val_dataset, val_dataset.get_batches(10)[0])
        # initialize constant loaders
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=8)

        f = open(outfile_name, 'w', newline='')
        writer = csv.writer(f)

        val_loss, val_acc = self._eval_epoch(val_loader, (writer, f))
        f.close()

        print("\nVal Loss: {}\tVal Acc: {}".format(
            np.round(val_loss, 5), np.round(val_acc * 100, 2)), flush=True)


    def _batch_loaders(self, dataset, k=None):
        batches = dataset.get_batches(k)
        loaders = []
        for subset_batch_indexes in batches:
            subset = Subset(dataset, subset_batch_indexes)
            loader = DataLoader(
                subset, batch_size=self.batch_size, shuffle=True,
                num_workers=8)
            loaders += [loader]
        return loaders

    def _format_model_subdir(self):
        subdir = "BMCB_mt{}vfd{}ss{}bhd{}cfd{}ks{}lr{}wp{}do{}nc{}wd{}bn{}pt{}mlr{}exmcb{}lmo{}nvf{}fe{}".\
                format(self.model_type, self.vis_feat_dim, self.spatial_size,
                       self.lm_hidden_dim, self.cmb_feat_dim,
                       self.kernel_size, self.learning_rate,
                       self.warmup_proportion, self.dropout, self.n_classes,
                       self.weight_decay, self.use_batchnorm, self.patience,
                       self.min_lr, self.use_external_MCB, self.lm_only,
                       self.normalize_vis_feats, self.freeze_epoch)
        return subdir

    def save(self):
        """
        Save model.

        Args
        ----
            models_dir: path to directory for saving NN models.

        """
        if (self.model is not None) and (self.save_dir is not None):

            if not os.path.isdir(os.path.join(self.save_dir, self.model_dir)):
                os.makedirs(os.path.join(self.save_dir, self.model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(self.save_dir, self.model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'trainer_dict': self.__dict__}, file)

    def load(self, model_dir, epoch, train_chunks=0, train_data_len=None):
        """
        Load a previously trained model.

        Args
        ----
            model_dir : directory where models are saved.
            epoch : epoch of model to load.

        """


        skip_list = ['vocab']

        epoch_file = "epoch_{}".format(epoch) + '.pth'
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_dict)
            else:
                checkpoint = torch.load(model_dict, map_location='cpu')

        for (k, v) in checkpoint['trainer_dict'].items():
            if k not in skip_list:
                setattr(self, k, v)

        self.USE_CUDA = torch.cuda.is_available()
        self._init_nn(train_chunks, train_data_len)
        self.model.load_state_dict(checkpoint['state_dict'])
        torch.set_rng_state(self.torch_rng_state)
        np.random.set_state(self.numpy_rng_state)
        self.nn_epoch += 1
