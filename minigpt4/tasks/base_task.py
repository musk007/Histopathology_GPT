"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
import wandb
from rouge_score import rouge_scorer
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.cfg = ""

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        self.cfg = cfg
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg
        print(f"run configurations:\n{cfg.run_cfg}")

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['test'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['test'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        answers = model(samples)['answers']
        results = self.metric_calculator(answers, samples['answer'], samples['image_id'])
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self, model, samples):
        with torch.no_grad():
            loss = model(samples)["loss"]
            predictions = model(samples)['answers']
            # for k in range (len(predictions)):
            #     print("\nPrediction\n")
            #     print("*"*50)
            #     print(predictions[k])
            #     print("*"*50,'\n\n')
                
            #     print("\nOriginal\n")
            #     print("*"*50)
            #     print(samples['answer'][k])
            #     print("*"*50,'\n\n')

        
            metrics = self.metric_calculator(predictions, samples['answer'], samples['image_id'])
            wandb.log({"Loss":loss})

        return loss, predictions, metrics

    def evaluate(self, model, data_loader, cuda_enabled=True):
        print("."*50)
        print("Evaluating")
        print("."*50)
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []
        for i in metric_logger.log_every(range(self.cfg.run_cfg.iters_per_epoch), 10, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= self.cfg.run_cfg.iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "num_iters_per_epoch": self.cfg.run_cfg.iters_per_epoch,
                    "iters": i,
                }
            )
            eval_output = self.inference_step(model=model, samples=samples)
            results.extend(eval_output)

        # for samples in metric_logger.log_every(data_loader, print_freq, header):
        #     samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            # eval_output = self.inference_step(model=model, samples=samples)
            # results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results
    
    def inference(self, model, data_loader, cuda_enabled=True):


        results = []
        # print("*"*50)
        # print(f"length of the dataloaders {len(data_loader)}")
        # print("*"*50)
        i = 0
        dataloader = next(data_loader)
        for d in dataloader:
            print(f"DataLoader within the loop is a dictionary with the following keys:\n{dataloader.keys()}")
            print(f"Iteration {i}\n")
            i += 1
            samples = iter(dataloader)
           
            # print("*"*50)
            # print(len(samples))
            # print("*"*50)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            eval_predictions, eval_metrics = self.inference_step(model=model, samples=samples)
            dit = {}
            dit['predictions'] = eval_predictions
            dit['metrics'] = eval_metrics
            results.extend(dit)

            if is_dist_avail_and_initialized():
                dist.barrier()

        return results
    
    def metric_calculator(self, predictions, labels, img_ids):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_tm = ROUGEScore()
        rouge1_recall = 0
        rouge1_precision = 0
        rouge1_f1 = 0

        rougeL_recall = 0
        rougeL_precision = 0
        rougeL_f1 = 0

        bleu_score_total = 0
        results = []
        dit = {}
        dit1 = {}

        for j in range(len(labels)):
            ## ROUGE scores
            # print(f"Image ID: {img_ids[j]}\n")
            # print(f"Prediction: \n{predictions[j]}\n")
            # print(f"Label: \n{labels[j]}\n")
            scores = scorer.score(labels[j],predictions[j])


            rouge1_precision+=scores['rouge1'][0]
            rouge1_recall+=scores['rouge1'][1]
            rouge1_f1+=scores['rouge1'][2]

            rougeL_precision+=scores['rougeL'][0]
            rougeL_recall+=scores['rougeL'][1]
            rougeL_f1+=scores['rougeL'][2]
            ## Torchmetrics
            tm_rouge = rouge_tm(predictions[j],labels[j])

            ## BLEU scores
            # Tokenize the sentences
            reference = labels[j].split()
            candidate = predictions[j].split()

            # Calculate BLEU score for this pair
            bleu_score = bleu([reference], candidate, smoothing_function=SmoothingFunction().method4,)
        
            # Add to total
            bleu_score_total += bleu_score
            # print(f"ROUGE1_precision Score is : {rouge1_precision/(j+1)}")
            # print(f"ROUGE1_recall Score is : {rouge1_recall/(j+1)}")
            # print(f"ROUGE1_f1 Score is : {rouge1_f1/(j+1)}\n")
            # print(f"ROUGEL_precision Score is : {rougeL_precision/(j+1)}")
            # print(f"ROUGEL_recall Score is : {rougeL_recall/(j+1)}")
            # print(f"ROUGEL_f1 Score is : {rougeL_f1/(j+1)}\n")
            # print(f"BLEU Score is : {bleu_score/(j+1)}\n")

        dit1["Precision"] = rouge1_precision / len(labels)
        dit1["Recall"] = rouge1_recall / len(labels)
        dit1["F1"] = rouge1_f1 / len(labels)
        dit['Rouge1'] = dit1
        wandb.log({"Rouge1_Precision": rouge1_precision / len(labels)})
        wandb.log({"Rouge1_Recall": rouge1_recall / len(labels)})
        wandb.log({"Rouge1_F1": rouge1_f1 / len(labels)})

        # print(f"ROUGE-1 Scores:\n")
        # print("Precision:\t\t{:.4f}".format(rouge1_precision / len(labels)))
        # print("Recall:\t\t{:.4f}".format(rouge1_recall / len(labels)))
        # print("F1:\t\t{:.4f}".format(rouge1_f1 / len(labels)))

        dit1 = {}
        dit1["Precision"] = rougeL_precision / len(labels)
        dit1["Recall"] = rougeL_recall / len(labels)
        dit1["F1"] = rougeL_f1 / len(labels)
        dit['RougeL'] = dit1
        wandb.log({"RougeL_Precision": rougeL_precision / len(labels)})
        wandb.log({"RougeL_Recall": rougeL_recall / len(labels)})
        wandb.log({"RougeL_F1": rougeL_f1 / len(labels)})

        # print(f"ROUGE-L Scores :\n")
        # print("Precision:\t\t{:.4f}".format(rougeL_precision / len(labels)))
        # print("Recall:\t\t{:.4f}".format(rougeL_recall / len(labels)))
        # print("F!:\t\t{:.4f}".format(rougeL_f1 / len(labels)))
        
        ### Torchmetrics
        dit["Torchmetrics"] = tm_rouge
        wandb.log({"Torchmetrics_ROUGE": tm_rouge})
        # print("Torchmetrics ROUGE Score:\n")
        # print("Results:\t\t{tm_rouge}")

        ### BLEU
        dit["BLEU"] = bleu_score_total / len(labels)
        average_bleu_score = bleu_score_total / len(labels)
        wandb.log({"BLEU": average_bleu_score})
        # print(f"Average BLEU Score: {average_bleu_score:.4f}")

        return dit
    
    

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
                # if self.cfg.wandb_log:
                if self.cfg.run_cfg.wandb_log:
                    wandb.log({"epoch": inner_epoch, "loss": loss})
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
