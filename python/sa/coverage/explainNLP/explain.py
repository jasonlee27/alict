import torch
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


class CasualInferenceExplain:
    def __init__(self, model, tokenizer, device, config):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.vocab_size = self.model.config.vocab_size
        self.base = 50
        self.threshold = config['threshold']
        # self.sos_token = self.model.config.sos_token
        # self.eos_token = self.model.config.eos_token

        self.mutate_rate = config['mutate_rate']
        self.mutate_num = config['mutate_num']

    @torch.no_grad()
    def predict_sentence(self, input_token):
        input_token = input_token.to(self.device)
        predict = self.model(input_token)
        logit = predict.logits
        output = logit.max(1)[1]
        return output

    def perturb_inputs(self, ori_token):
        input_length = ori_token.shape[1]
        mutated_x = ori_token.repeat((self.mutate_num, 1)).to(self.device)
        random_x = (torch.rand([self.mutate_num, input_length], device=self.device) * self.vocab_size).int()
        mask = (torch.rand([self.mutate_num, input_length], device=self.device) < self.mutate_rate).float()
        mask[:, 0] = 1
        mask[:, -1] = 1
        mutated_x = random_x * (1 - mask) + mutated_x * mask
        mutated_x = mutated_x.to(torch.int64).to(self.device)
        return mutated_x, mask

    def compute_weights(self, mask, mutated_y):
        m = LogisticRegression(fit_intercept=True)
        m.fit(mask.detach().cpu().numpy(), mutated_y.reshape(-1).detach().cpu().numpy())
        weight = m.coef_
        return weight.reshape([-1])

    def visualization(self, input_token, important_score):
        res = []
        input_token = input_token.reshape([-1])
        for i, (tk, s) in enumerate(zip(input_token, important_score)):
            tk_str = self.tokenizer.decode(tk)
            res.append([i, int(tk), tk_str, s])
        res = sorted(res, key=lambda t: t[3], reverse=True)
        return res

    def explain(self, input_sentence):
        input_token = self.tokenizer(input_sentence, return_tensors="pt", padding=True).input_ids
        ori_predict = self.predict_sentence(input_token)
        mutated_x, mask = self.perturb_inputs(input_token)
        mutated_y = []

        iter_num = self.mutate_num // self.base
        if iter_num * self.base != self.mutate_num:
            iter_num = iter_num + 1
        for i in range(iter_num):
            st, ed = self.base * i, min(self.base * (i + 1), len(mutated_x))
            x = mutated_x[st:ed]
            y = self.predict_sentence(x)
            mutated_y.append(y)
        mutated_y = torch.cat(mutated_y)
        mutated_y = (mutated_y == ori_predict)
        important_score = self.compute_weights(mask, mutated_y)
        res = self.visualization(input_token, important_score)

        return res, ori_predict

    def evaluate(self, mask: torch.tensor, ori_pred, mask_pos):
        new_tensor = mask.repeat(500, 1).to(self.device)
        rand_tensor = (torch.rand(new_tensor[:, mask_pos].shape, device=self.device) * 10000).int()
        rand_tensor = rand_tensor.to(torch.long)
        new_tensor[:, mask_pos] = rand_tensor

        new_pred = self.predict_sentence(new_tensor)
        return new_pred.eq(ori_pred).sum() / 500

    def create_mask(self, res, input_sentence, orig_pred):
        reverse_res = list(reversed(res))
        input_token = self.tokenizer(input_sentence, return_tensors="pt", padding=True).input_ids
        pad_id = self.tokenizer.pad_token_id
        unk_id = self.tokenizer.unk_token_id

        mask = input_token.clone()
        negative = [r for r in res if r[-1] < 0 and r[1] not in self.tokenizer.all_special_ids]
        mask_pos = []
        for i in range(len(negative)):
            pos, tk = negative[i][0], negative[i][1]
            mask[:, pos] = unk_id
            mask_pos.append(pos)
        for i in range(len(negative)):
            pos, tk = negative[i][0], negative[i][1]
            score = self.evaluate(mask, orig_pred, mask_pos)
            if score > self.threshold:
                break
            else:
                mask[:, pos] = tk
        return mask

