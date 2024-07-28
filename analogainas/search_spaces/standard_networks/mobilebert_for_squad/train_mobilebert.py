import wandb
import torch
from aihwkit.simulator.presets.inference import StandardHWATrainingPreset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
import yaml
from utils import to_list, load_and_cache_examples, evaluate, train_epoch, load_model_tokenizer, load_dataloader_examples_features

# Initialize W&B and configuration
wandb.init(project="mobilebert_squadv1")

# Define RPU Config
rpu_config = StandardHWATrainingPreset()

# Load configuration
with open('config/configuration.yaml') as f:
    sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)

# Main training function
def main(t_inferences=[0., 3600., 86400.], n_reps=5, early_exit_n_iters=-1):
    max_seq_length = wandb.config.max_seq_length
    logging_step_frequency = wandb.config.logging_step_frequency
    batch_size_train = wandb.config.batch_size_train
    batch_size_eval = wandb.config.batch_size_eval
    weight_decay = wandb.config.weight_decay
    num_training_epochs = wandb.config.num_training_epochs
    learning_rate = 10 ** -wandb.config.learning_rate
    model_id = "csarron/mobilebert-uncased-squad-v1" # MobileBERT fine-tuned on SQuAD v1

    model, tokenizer = load_model_tokenizer(model_id, "data")
    train_dataloader, train_examples, train_features = load_dataloader_examples_features(model_id, tokenizer, evaluate=False, batch_size=batch_size_train, max_seq_length=max_seq_length)
    test_dataloader, test_examples, test_features = load_dataloader_examples_features(model_id, tokenizer, evaluate=True, batch_size=batch_size_eval, max_seq_length=max_seq_length)

    model = convert_to_analog(model, rpu_config, verbose=False)

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AnalogAdam(optimizer_grouped_parameters, lr=learning_rate)
    t_total = len(train_dataloader) // num_training_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    model.zero_grad()
    for current_epoch in range(0, num_training_epochs):
        print("Training epoch: ", current_epoch)
        model.train()
        train_epoch(train_dataloader, model, optimizer, scheduler, current_epoch, logging_step_frequency, wandb_logging=True, early_exit_n_iters=early_exit_n_iters)
        with torch.no_grad():
            model.eval()
            for t in t_inferences:
                print('t_inference:', t)
                f1_scores = []
                for rep in range(n_reps):
                    model.drift_analog_weights(t)
                    result = evaluate(model, tokenizer, test_examples, test_features, test_dataloader, cache_dir='data', early_exit_n_iters=early_exit_n_iters)
                    f1_scores.append(result['f1'])
                print("=====", t, np.mean(f1_scores), np.std(f1_scores))

# Run the optimization loop
sweep_id = wandb.sweep(sweep=sweep_configuration, project="mobilebert_squadv1")
wandb.agent(sweep_id, function=main, count=10)

