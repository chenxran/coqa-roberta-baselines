import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils.eval_utils import compute_eval_metric
from models.layers import multi_nll_loss
from utils import constants as Constants
from models.bert import BertForCoQA


class Model(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, config, train_set=None):
        # Book-keeping.
        self.config = config
        # if self.config['pretrained']:
        #     self.init_saved_network(self.config['pretrained'])
        # else:
        #     assert train_set is not None
        #     print('Train vocab: {}'.format(len(train_set.vocab)))
        #     vocab = Counter()
        #     for w in train_set.vocab:
        #         if train_set.vocab[w] >= config['min_freq']:
        #             vocab[w] = train_set.vocab[w]
        #     print('Pruned train vocab: {}'.format(len(vocab)))
        #     # Building network.
        #     word_model = WordModel(embed_size=self.config['embed_size'],
        #                            filename=self.config['embed_file'],
        #                            embed_type =self.config['embed_type'],
        #                            top_n=self.config['top_vocab'],
        #                            additional_vocab=vocab)
        #     self.config['embed_size'] = word_model.embed_size
        #     self._init_new_network(train_set, word_model)
        self.network = BertForCoQA(config)
        
        # num_params = 0
        # for name, p in self.network.named_parameters():
        #     print('{}: {}'.format(name, str(p.size())))
        #     num_params += p.numel()
        # print('#Parameters = {}\n'.format(num_params))

        self._init_optimizer()

    def _init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.config['learning_rate'],
                                       momentum=self.config['momentum'],
                                       weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.config['weight_decay'])
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(parameters, self.config['learning_rate'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.config['optimizer'])

    def predict(self, step, ex, update=True, out_predictions=False):
        # Train/Eval mode
        self.network.train(update)

        # Run forward
        res = self.network(ex)

        score_s, score_e = res['score_s'], res['score_e']

        output = {
            'f1': 0.0,
            'em': 0.0,
            'loss': 0.0
        }

        # Loss cannot be computed for test-time as we may not have targets
        if update:
            # Compute loss and accuracies
            # loss = self.compute_span_loss(score_s, score_e, res['targets'])
            loss = res['loss'] / self.config['gradient_accumulation_steps']
            output['loss'] = loss.cpu().float()

            # Clear gradients and run backward
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if (step > 0) and (step % self.config['gradient_accumulation_steps'] == 0):
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['grad_clipping'])

                # Update parameters
                self.optimizer.step()

        # 现在的问题在于，我不知道这个target对应的是不是真的就是target。因此我可能需要把span_text拿出来和这个对比看看，到底对不对
        if (not update) or self.config['predict_train']:
            predictions, spans = self.extract_predictions(ex, score_s, score_e)
            output['f1'], output['em'] = self.evaluate_predictions(predictions, ex['answers'])
            if out_predictions:
                output['predictions'] = predictions
                output['spans'] = spans

        del ex

        return output

    # def compute_span_loss(self, score_s, score_e, targets):
    #     assert targets.size(0) == score_s.size(0) == score_e.size(0)
    #     if self.config['sum_loss']:
    #         loss = multi_nll_loss(score_s, targets[:, :, 0]) + multi_nll_loss(score_e, targets[:, :, 1])
    #     else:
    #         loss = F.nll_loss(score_s, targets[:, 0]) + F.nll_loss(score_e, targets[:, 1])
    #     return loss

    def extract_predictions(self, ex, score_s, score_e):
        # Transfer to CPU/normal tensors for numpy ops (and convert log probabilities to probabilities)
        score_s = score_s.exp().squeeze()
        score_e = score_e.exp().squeeze()

        predictions = []
        spans = []
        for i, (_s, _e) in enumerate(zip(score_s, score_e)):
            if self.config['predict_raw_text']:
                prediction, span = self._scores_to_raw_text(ex['raw_evidence_text'][i], ex['length'][i],
                                                            ex['offsets'][i], _s, _e)
            else:
                prediction, span = self._scores_to_text(ex['evidence_text'][i], _s, _e)
            predictions.append(prediction)
            spans.append(span)
        return predictions, spans

    # TODO: s_idx, e_idx与question_length的一些处理问题
    def _scores_to_text(self, text, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        scores = torch.ger(score_s.squeeze(), score_e.squeeze())
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return ' '.join(text[s_idx: e_idx + 1]), (int(s_idx), int(e_idx))

    def _scores_to_raw_text(self, raw_text, length, offsets, score_s, score_e):
        max_len = self.config['max_answer_len'] or score_s.size(1)
        score_s = score_s.squeeze()
        score_e = score_e.squeeze()
        scores = torch.ger(score_s[length[0] + 2: length[1] - 1], score_e[length[0] + 2: length[1] - 1])  # TODO: Roberta/Bert need changes here
        scores.triu_().tril_(max_len - 1)
        scores = scores.cpu().detach().numpy()
        s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
        return raw_text[offsets[s_idx][0]: offsets[e_idx][1]], (offsets[s_idx][0], offsets[e_idx][1])

    def evaluate_predictions(self, predictions, answers):
        f1_score = compute_eval_metric('f1', predictions, answers)
        em_score = compute_eval_metric('em', predictions, answers)
        return f1_score, em_score

    def save(self, dirname):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'config': self.config,
            'dir': dirname,
        }
        try:
            torch.save(params, os.path.join(dirname, Constants._SAVED_WEIGHTS_FILE))
        except BaseException:
            print('[ WARN: Saving failed... continuing anyway. ]')
