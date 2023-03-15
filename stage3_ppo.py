import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from copy import deepcopy

import pandas as pd
import torch
from cores.nn import BLOOMActor, BLOOMCritic, GPTActor, GPTCritic, OPTActor, OPTCritic, RewardModel
from cores.trainer import PPOTrainer
from cores.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from torch.optim import Adam
from transformers import AutoTokenizer, BloomTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

from colossalai.nn.optimizer import HybridAdam


def main(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda', initial_scale=2 ** 5)
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')

    # configure model
    with strategy.model_init_context():
        # if torch.cuda.is_available():
        #     actor = GPTActor(pretrained=args.pretrain).to("cuda")
        #     critic = GPTCritic().to("cuda")
        # elif torch.backends.mps.is_available():
        #     actor = GPTActor(pretrained=args.pretrain).to("mps")
        #     critic = GPTCritic().to("mps")
        # else:
        actor = GPTActor(pretrained=args.pretrain)
        critic = GPTCritic()

        initial_model = deepcopy(actor)
        reward_model = RewardModel(deepcopy(critic.model), deepcopy(critic.value_head))
        # if torch.cuda.is_available():
        #     reward_model = reward_model.to("cuda")
        # elif torch.backends.mps.is_available():
        #     reward_model = reward_model.to("mps")
        # else:
        #     pass

    # configure optimizer
    if args.strategy.startswith('colossalai'):
        actor_optim = HybridAdam(actor.parameters(), lr=5e-6)
        critic_optim = HybridAdam(critic.parameters(), lr=5e-6)
    else:
        actor_optim = Adam(actor.parameters(), lr=5e-6)
        critic_optim = Adam(critic.parameters(), lr=5e-6)

    # configure tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrain)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = pd.read_csv(args.prompt_path)['prompt']

    def tokenize_fn(texts):
        batch = tokenizer(texts, return_tensors='pt', max_length=96, padding=True, truncation=True)
        # if torch.cuda.is_available():
        #     return {k: v.to("cuda") for k, v in batch.items()}
        # elif torch.backends.mps.is_available():
        #     return {k: v.to("mps") for k, v in batch.items()}
        # else:
        return {k: v for k, v in batch.items()}

    (actor, actor_optim), (critic, critic_optim), reward_model, initial_model = strategy.prepare(
        (actor, actor_optim), (critic, critic_optim), reward_model, initial_model)

    # configure trainer
    trainer = PPOTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        actor_optim,
        critic_optim,
        max_epochs=args.max_epochs,
        train_batch_size=args.train_batch_size,
        experience_batch_size=args.experience_batch_size,
        tokenizer=tokenize_fn,
        max_length=128,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    trainer.fit(dataset,
                num_episodes=args.num_episodes,
                max_timesteps=args.max_timesteps,
                update_timesteps=args.update_timesteps)
    # save model checkpoint after fitting on only rank0
    strategy.save_model(actor, os.path.join(args.out_path, 'actor_checkpoint_prompts.pt'), only_rank0=True)
    # save optimizer checkpoint on all ranks
    strategy.save_optimizer(actor_optim, os.path.join(args.out_path, 'actor_optim_checkpoint_prompts.pt'),
                            only_rank0=False)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default=os.path.join(BASE_DIR, "data/prompt_train.csv"))
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2'],
                        default='naive')
    parser.add_argument('--pretrain', type=str, default=os.path.join(BASE_DIR, "models/base"))
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--max_timesteps', type=int, default=10)
    parser.add_argument('--update_timesteps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--experience_batch_size', type=int, default=8)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--out_path', type=str, default=os.path.join(BASE_DIR, "models/checkpoints"))
    args = parser.parse_args()
    main(args)
