import os
import torch
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(model_name_or_path, tokenizer, max_seq_length, evaluate=False, output_examples=False, overwrite_cache=False, cache_dir="data"):
    cached_features_file = os.path.join(
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
        ),
    )
    if os.path.exists(cached_features_file) and not overwrite_cache:
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        import tensorflow_datasets as tfds
        tfds_examples = tfds.load("squad", data_dir=cache_dir)
        examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=128,
            max_query_length=64,
            is_training=not evaluate,
            return_dataset="pt",
            threads=8,
        )
        torch.save(
            {"features": features, "dataset": dataset, "examples": examples},
            cached_features_file,
        )
    if output_examples:
        return dataset, examples, features
    return dataset

def evaluate(model, tokenizer, examples, features, eval_dataloader, cache_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), early_exit_n_iters=-1):
    all_results = []
    batch_idx = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs.to_tuple()]
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]
                result = SquadResult(unique_id, start_logits, end_logits, start_top_index=start_top_index, end_top_index=end_top_index, cls_logits=cls_logits)
            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)
        if batch_idx == early_exit_n_iters:
            break
        batch_idx += 1

    output_prediction_file = os.path.join(cache_dir, "predictions.json")
    output_nbest_file = os.path.join(cache_dir, "nbest_predictions.json")
    predictions = compute_predictions_logits(examples[0:early_exit_n_iters], features[0:early_exit_n_iters], all_results[0:early_exit_n_iters], 20, 30, True, output_prediction_file, output_nbest_file, None, False, False, 0.0, tokenizer)
    results = squad_evaluate(examples[0:early_exit_n_iters], predictions)
    return results

def train_epoch(train_dataloader, model, optimizer, scheduler, current_epoch, logging_step_frequency, wandb_logging=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), early_exit_n_iters=-1):
    model.train()
    step = 0
    n_steps = len(train_dataloader)
    with tqdm(train_dataloader, desc="Iteration") as tepoch:
        for batch in tepoch:
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "start_positions": batch[3], "end_positions": batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad(set_to_none=True)
            if step % logging_step_frequency == 0:
                tepoch.set_postfix(loss=loss.item(), lr=scheduler.get_lr()[0])
                if wandb_logging:
                    wandb.log({"step": n_steps * current_epoch + step, "training_loss": loss.item()})
            if step == early_exit_n_iters:
                break
            step += 1

def load_model_tokenizer(model_id, cache_dir="cache", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=True, cache_dir=cache_dir, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id, from_tf=False, config=config, cache_dir=cache_dir)
    model = model.to(device)
    return model, tokenizer

def load_dataloader_examples_features(model_id, tokenizer, evaluate, batch_size=16, max_seq_length=320):
    dataset, examples, features = load_and_cache_examples(model_id, tokenizer, max_seq_length, evaluate=evaluate, output_examples=True)
    if evaluate:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader, examples, features

