import numpy as np
import torch
import torch.nn as nn
import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
import time
import math


class Attacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack(self, imgs, txts, txt2img, all_texts, device='cpu', max_length=30, scales=None, masks=None, **kwargs):
        with torch.no_grad():
            img_input = self.img_attacker.normalization(imgs)
            if hasattr(self.model, 'inference_image'):
                origin_img_output = self.model.inference_image(img_input)
                img_feat = origin_img_output['image_feat']
            else:
                img_feat = self.model.encode_image(img_input)

            img_supervisions = img_feat[txt2img]

        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)

            if hasattr(self.model, 'inference_text'):
                txts_output = self.model.inference_text(txts_input)
                txt_supervisions = txts_output['text_feat']
            else:
                txt_supervisions = self.model.encode_text(txts_input.input_ids)

            all_texts_input = self.txt_attacker.tokenizer(all_texts, padding='max_length', truncation=True,
                                                          max_length=max_length, return_tensors="pt").to(device)

            if hasattr(self.model, 'inference_text'):
                all_texts_output = self.model.inference_text(all_texts_input)
                all_txt_supervisions = all_texts_output['text_feat']
            else:
                all_txt_supervisions = self.model.encode_text(all_texts_input.input_ids)

        start_time = time.time()
        adv_imgs, last_adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, all_txt_supervisions,
                                                                      device,
                                                                      scales=scales, txt_embeds=txt_supervisions)
        end_time = time.time()
        execuate_time = end_time - start_time

        with torch.no_grad():
            adv_img_input = self.img_attacker.normalization(adv_imgs)
            last_adv_img_input = self.img_attacker.normalization(last_adv_imgs)

            if hasattr(self.model, 'inference_image'):
                adv_imgs_outputs = self.model.inference_image(adv_img_input)
                adv_img_feat = adv_imgs_outputs['image_feat']

                last_adv_imgs_outputs = self.model.inference_image(last_adv_img_input)
                last_adv_img_feat = last_adv_imgs_outputs['image_feat']
            else:
                adv_img_feat = self.model.encode_image(adv_img_input)
                last_adv_img_feat = self.model.encode_image(last_adv_img_input)

            adv_img_supervisions = adv_img_feat[txt2img]
            last_adv_img_supervisions = last_adv_img_feat[txt2img]

        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
                                                       adv_img_embeds=adv_img_supervisions,
                                                       last_adv_img_embeds=last_adv_img_supervisions)
        return adv_imgs, adv_txts, execuate_time


class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, sample_numbers=5,
                 momentum=1.0,
                 nge_beta=1.5, nge_num_samples=20,
                 adaptive_weights=True,
                 pz_enabled=True, pz_prob=1.0,
                 pz_region_min=0.5, pz_region_max=0.9,
                 pz_resize_min=0.8, pz_resize_max=1.2,
                 pz_block_count=1):

        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers
        self.momentum = momentum
        self.nge_num_samples = nge_num_samples
        self.nge_beta = nge_beta
        self.adaptive_weights = adaptive_weights

        self.pz_enabled = pz_enabled
        self.pz_prob = pz_prob
        self.pz_region_min = pz_region_min
        self.pz_region_max = pz_region_max
        self.pz_resize_min = pz_resize_min
        self.pz_resize_max = pz_resize_max
        self.pz_block_count = int(pz_block_count)

    def _apply_pz(self, x, num_transforms):
        if not self.pz_enabled:
            return x.repeat(num_transforms, 1, 1, 1)
        b, c, h, w = x.shape
        transformed_versions = []
        for _ in range(num_transforms):
            if torch.rand(1).item() > self.pz_prob:
                transformed_versions.append(x.clone())
                continue
            x_clone = x.clone()

            for _ in range(self.pz_block_count):
                region_scale_h = random.uniform(self.pz_region_min, self.pz_region_max)
                region_scale_w = random.uniform(self.pz_region_min, self.pz_region_max)
                region_h, region_w = int(h * region_scale_h), int(w * region_scale_w)

                if region_h <= 0: region_h = 1
                if region_w <= 0: region_w = 1

                top, left = random.randint(0, h - region_h), random.randint(0, w - region_w)
                region = x_clone[..., top:top + region_h, left:left + region_w]

                pz_resize_rate = random.uniform(self.pz_resize_min, self.pz_resize_max)

                img_resize = int(region_h * pz_resize_rate), int(region_w * pz_resize_rate)
                if img_resize[0] <= 0 or img_resize[1] <= 0: continue

                rescaled = F.interpolate(region, size=img_resize, mode='bicubic', align_corners=False)
                h_rem = abs(img_resize[0] - region_h)
                w_rem = abs(img_resize[1] - region_w)
                pad_top = torch.randint(0, h_rem + 1, (1,)).item()
                pad_bottom = h_rem - pad_top
                pad_left = torch.randint(0, w_rem + 1, (1,)).item()
                pad_right = w_rem - pad_left
                padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)
                transformed_region = F.interpolate(padded, size=(region_h, region_w), mode='bicubic',
                                                   align_corners=False)
                x_clone[..., top:top + region_h, left:left + region_w] = transformed_region
            transformed_versions.append(x_clone)
        return torch.cat(transformed_versions, 0)

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img, all_txt_supervisions):
        device = adv_imgs_embeds.device
        U, S, V = torch.svd(all_txt_supervisions.T.to(torch.float32))
        projection_matrix = U[:, 1:len(U)] @ U[:, 1:len(U)].t()
        adv_imgs_embeds = adv_imgs_embeds @ projection_matrix
        txts_embeds = txts_embeds @ projection_matrix
        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        return loss_IaTcpos

    def rand3Num(self):
        while True:
            num1 = random.randint(1, 100)
            if 100 - num1 > 1:
                num2 = random.randint(1, 100 - num1)
            else:
                num1 = 98
                num2 = 1
            num3 = 100 - num1 - num2
            if 1 <= num3 <= 100 and num1 < num3 and num3 < num2:
                break
        return (num1, num2, num3)

    def _get_nge_grad(self, model, adv_imgs, txt_embeds, txt2img, all_txt_supervisions, device):
        def _calculate_single_grad(imgs_input):
            original_b = imgs_input.shape[0]
            imgs_input.requires_grad_()
            pz_list = [imgs_input]
            target_expand_times = 2

            for _ in range(target_expand_times):
                single_pz = self._apply_pz(imgs_input, num_transforms=1)
                pz_list.append(single_pz)
            pz_imgs = torch.cat(pz_list, dim=0)
            scaled_imgs = self.get_scaled_imgs(pz_imgs, [0.5, 0.75, 1.25, 1.5], device)
            img_input_norm = self.normalization(scaled_imgs) if self.normalization is not None else scaled_imgs
            if hasattr(model, 'inference_image'):
                adv_imgs_output = model.inference_image(img_input_norm)
                adv_imgs_embeds = adv_imgs_output['image_feat']
            else:
                adv_imgs_embeds = model.encode_image(img_input_norm)
            model.zero_grad()
            with torch.enable_grad():
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                num_total_transforms = scaled_imgs.shape[0] // original_b
                for i in range(num_total_transforms):
                    start_idx = i * original_b
                    end_idx = (i + 1) * original_b
                    loss_item = self.loss_func(adv_imgs_embeds[start_idx:end_idx], txt_embeds, txt2img,
                                               all_txt_supervisions)
                    loss += loss_item

                loss = loss / num_total_transforms
            loss.backward()
            grad = imgs_input.grad.clone()
            imgs_input.grad.zero_()
            model.zero_grad()
            return grad

        base_grad = _calculate_single_grad(adv_imgs.detach().clone())
        nge_grads = [base_grad]
        bound = self.eps * self.nge_beta
        for _ in range(self.nge_num_samples - 1):
            neighbor_imgs = adv_imgs.detach() + torch.randn_like(adv_imgs) * bound
            neighbor_imgs.clamp_(0.0, 1.0)
            neighbor_grad = _calculate_single_grad(neighbor_imgs)
            nge_grads.append(neighbor_grad)
        avg_grad = torch.stack(nge_grads, dim=0).mean(dim=0)
        return avg_grad

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs
        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        reverse_transform = transforms.Resize(ori_shape, interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]), int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape, interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            reversed_imgs = reverse_transform(scaled_imgs)
            result.append(reversed_imgs)
        return torch.cat([imgs, ] + result, 0)


    def _compute_adaptive_weights(self, loss_history):
        if not self.adaptive_weights or len(loss_history) < 2:
            return (20, 50, 30)

        recent_losses = loss_history[-3:]

        if len(recent_losses) == 2:
            return (20, 50, 30)
        else:
            loss_trend = recent_losses[-1] - recent_losses[1]
            loss_variation = abs(recent_losses[1] - recent_losses[0])

        if loss_variation < 1e-9:
            norm_trend = 0.0
        else:
            norm_trend = math.tanh(loss_trend / (loss_variation + 1e-9))

        base_current = 20
        base_past = 30
        swing = 5 * norm_trend

        w_current = base_current + swing
        w_past = base_past - swing
        w_orig = 100 - w_current - w_past

        if w_orig < 5:
            w_orig = 5
            total_other = w_current + w_past
            if total_other == 0: total_other = 1
            w_current = int((100 - w_orig) * (w_current / total_other))
            w_past = 100 - w_orig - w_current

        w_current = max(1, w_current)
        w_past = max(1, w_past)

        current_sum = w_current + w_orig + w_past
        if current_sum != 100:
            w_current += (100 - current_sum)

        return (w_current, w_orig, w_past)


    def txt_guided_attack(self, model, imgs, txt2img, all_txt_supervisions, device, scales=None, txt_embeds=None):
        model.eval()
        b, _, _, _ = imgs.shape

        loss_history = []

        if self.adaptive_weights:
            with torch.no_grad():
                img_input_norm = self.normalization(imgs) if self.normalization is not None else imgs
                if hasattr(model, 'inference_image'):
                    clean_output = model.inference_image(img_input_norm)
                    clean_embeds = clean_output['image_feat']
                else:
                    clean_embeds = model.encode_image(img_input_norm)

                clean_loss = self.loss_func(clean_embeds, txt_embeds, txt2img, all_txt_supervisions)
                loss_history.append(clean_loss.item())

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        last_adv_imgs = None

        momentum_grad = torch.zeros_like(imgs, device=device)

        for step in range(self.steps):
            if last_adv_imgs is not None:
                clone_adv_imgs = adv_imgs.clone()

                if self.adaptive_weights:
                    best_weights = self._compute_adaptive_weights(loss_history)

                    adv_imgs = (best_weights[0] / 100) * clone_adv_imgs + \
                               (best_weights[1] / 100) * imgs + \
                               (best_weights[2] / 100) * last_adv_imgs

                    with torch.no_grad():
                        img_input_norm = self.normalization(adv_imgs) if self.normalization is not None else adv_imgs
                        if hasattr(model, 'inference_image'):
                            temp_output = model.inference_image(img_input_norm)
                            temp_embeds = temp_output['image_feat']
                        else:
                            temp_embeds = model.encode_image(img_input_norm)

                        current_loss = self.loss_func(temp_embeds, txt_embeds, txt2img, all_txt_supervisions)
                        loss_history.append(current_loss.item())

                else:
                    loss_list = []
                    samples = [self.rand3Num() for _ in range(self.sample_numbers)]
                    for sample in samples:
                        temp_adv_imgs = (sample[0] / 100) * clone_adv_imgs + (sample[1] / 100) * imgs + (
                                    sample[2] / 100) * last_adv_imgs


                final_grad = self._get_nge_grad(model, adv_imgs, txt_embeds, txt2img, all_txt_supervisions, device)
                update_grad = final_grad
                norm_update_grad = update_grad / torch.mean(torch.abs(update_grad), dim=(1, 2, 3), keepdim=True)
                momentum_grad = self.momentum * momentum_grad + norm_update_grad
                perturbation = self.step_size * momentum_grad.sign()
                adv_imgs = clone_adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps).clamp(0.0, 1.0)
                last_adv_imgs = clone_adv_imgs.clone()

            else:

                if self.adaptive_weights:
                    with torch.no_grad():
                        img_input_norm = self.normalization(adv_imgs) if self.normalization is not None else adv_imgs
                        if hasattr(model, 'inference_image'):
                            temp_output = model.inference_image(img_input_norm)
                            temp_embeds = temp_output['image_feat']
                        else:
                            temp_embeds = model.encode_image(img_input_norm)

                        init_loss = self.loss_func(temp_embeds, txt_embeds, txt2img, all_txt_supervisions)
                        loss_history.append(init_loss.item())

                last_adv_imgs = adv_imgs.clone()
                final_grad = self._get_nge_grad(model, adv_imgs, txt_embeds, txt2img, all_txt_supervisions, device)

                update_grad = final_grad
                norm_update_grad = update_grad / torch.mean(torch.abs(update_grad), dim=(1, 2, 3), keepdim=True)
                momentum_grad = self.momentum * momentum_grad + norm_update_grad
                perturbation = self.step_size * momentum_grad.sign()

                adv_imgs = adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps).clamp(0.0, 1.0)

        return adv_imgs, last_adv_imgs

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs
        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        reverse_transform = transforms.Resize(ori_shape, interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]), int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape, interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            reversed_imgs = reverse_transform(scaled_imgs)
            result.append(reversed_imgs)
        return torch.cat([imgs, ] + result, 0)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']

filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2], adaptive_weights=True):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios
        self.adaptive_weights = adaptive_weights

        print(f">> TextAttacker Initialized. Adaptive Weights: {self.adaptive_weights}")


    def _compute_adaptive_weights(self, loss_history):

        if not self.adaptive_weights or len(loss_history) < 2:
            return (20, 60, 20)

        recent_losses = loss_history[-3:]

        if len(recent_losses) == 2:
            return (20, 60, 20)
        else:
            loss_trend = recent_losses[-1] - recent_losses[1]
            loss_variation = abs(recent_losses[1] - recent_losses[0])

        if loss_variation < 1e-9:
            norm_trend = 0.0
        else:
            norm_trend = math.tanh(loss_trend / (loss_variation + 1e-9))

        base_current = 20
        base_past = 20
        swing = 5 * norm_trend

        w_current = base_current + swing
        w_past = base_past - swing

        w_orig = 100 - w_current - w_past

        w_current = max(1, w_current)
        w_past = max(1, w_past)
        w_orig = 100 - w_current - w_past

        return (w_current, w_orig, w_past)


    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, last_adv_img_embeds=None):
        device = self.ref_net.device
        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits

        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        if hasattr(net, 'inference_text'):
            origin_output = net.inference_text(text_inputs)
            if self.cls:
                origin_embeds = origin_output['text_feat'][:, 0, :].detach()
            else:
                origin_embeds = origin_output['text_feat'].flatten(1).detach()
        else:
            origin_embeds = net.encode_text(text_inputs.input_ids).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            loss_history = []

            clean_emb_single = origin_embeds[i].unsqueeze(0)

            if self.adaptive_weights:
                if adv_img_embeds is None:
                    init_clean_loss = self.loss_func(clean_emb_single, img_embeds, i)
                else:
                    l_orig = 0.6 * self.loss_func(clean_emb_single, img_embeds, i)
                    l_curr = 0.2 * self.loss_func(clean_emb_single, adv_img_embeds, i)
                    l_past = 0.2 * self.loss_func(clean_emb_single, last_adv_img_embeds, i)
                    init_clean_loss = l_orig + l_curr + l_past

                loss_history.append(init_clean_loss.item())

            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)
            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break
                tgt_word = words[top_index[0]]
                if tgt_word in filter_words or keys[top_index[0]][0] > self.max_length - 2:
                    continue
                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = self.get_substitues(substitutes, 1, word_pred_scores, self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_
                    if substitute == tgt_word or '##' in substitute or substitute in filter_words:
                        continue
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))

                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)

                if hasattr(net, 'inference_text'):
                    replace_output = net.inference_text(replace_text_input)
                    if self.cls:
                        replace_embeds = replace_output['text_feat'][:, 0, :]
                    else:
                        replace_embeds = replace_output['text_feat'].flatten(1)
                else:
                    replace_embeds = net.encode_text(replace_text_input.input_ids)

                if adv_img_embeds is None:
                    loss = self.loss_func(replace_embeds, img_embeds, i)
                else:
                    w_curr_raw, w_orig_raw, w_past_raw = self._compute_adaptive_weights(loss_history)

                    lambda_curr = w_curr_raw / 100.0
                    lambda_orig = w_orig_raw / 100.0
                    lambda_past = w_past_raw / 100.0

                    loss = lambda_orig * self.loss_func(replace_embeds, img_embeds, i) + \
                           lambda_curr * self.loss_func(replace_embeds, adv_img_embeds, i) + \
                           lambda_past * self.loss_func(replace_embeds, last_adv_img_embeds, i)

                best_loss_val = loss.max().item()
                loss_history.append(best_loss_val)

                candidate_idx = loss.argmax()
                final_words[top_index[0]] = available_substitutes[candidate_idx]
                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1
            final_adverse.append(' '.join(final_words))
        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def _tokenize(self, text):
        words = text.split(' ')
        sub_words, keys, index = [], [], 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        masked_words = [words[0:i] + ['[UNK]'] + words[i + 1:] for i in range(len(words))]
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device
        masked_texts = [' '.join(words) for words in self._get_masked(text)]
        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)

            if hasattr(net, 'inference_text'):
                masked_output = net.inference_text(masked_text_input)
                if self.cls:
                    masked_embed = masked_output['text_feat'][:, 0, :].detach()
                else:
                    masked_embed = masked_output['text_feat'].flatten(1).detach()
            else:
                masked_embed = net.encode_text(masked_text_input.input_ids).detach()

            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)
        criterion = torch.nn.KLDivLoss(reduction='none')
        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))
        return import_scores.sum(dim=-1)

    def get_substitues(self, substitutes, use_bpe, substitutes_score=None, threshold=3.0):
        words = []
        sub_len, k = substitutes.size()
        if sub_len == 0:
            return words
        elif sub_len == 1:
            for (i, j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(self.tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes)
        return words

    def get_bpe_substitues(self, substitutes):
        device = self.ref_net.device
        substitutes = substitutes[0:12, 0:4]
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if not all_substitutes:
                all_substitutes = [[int(c)] for c in substitutes[i]]
            else:
                new_all_substitutes = [all_sub + [int(j)] for all_sub in all_substitutes for j in substitutes[i]]
                all_substitutes = new_all_substitutes

        c_loss = nn.CrossEntropyLoss(reduction='none')
        if not all_substitutes: return []

        all_substitutes_tensor = torch.tensor(all_substitutes, device=device)[:24]
        if all_substitutes_tensor.numel() == 0: return []

        N, L = all_substitutes_tensor.size()
        word_predictions = self.ref_net(all_substitutes_tensor)[0]
        ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes_tensor.view(-1))
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
        _, word_indices = torch.sort(ppl)
        word_list = [all_substitutes_tensor[i] for i in word_indices]

        final_words = []
        for word in word_list:
            tokens = [self.tokenizer._convert_id_to_token(int(i)) for i in word]
            text = self.tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words