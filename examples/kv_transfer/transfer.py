#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Round Conversation Performance Test for vLLM Instances

This script measures and compares performance metrics (TTFT, generation time, etc.) 
between two vLLM instances through multi-round conversations.

Features:
- Tests two vLLM instances with multi-turn conversations
- Uses local ShareGPT dataset for realistic multi-round dialogues
- Instance 1: conversations in original order
- Instance 2: same conversations but in random shuffled order
- Uses local model files (no remote API calls)
- Connects to locally running vLLM instances via their OpenAI-compatible API endpoints
- Generates detailed per-turn and aggregate performance statistics

Workflow:
1. Load multi-round conversations from ShareGPT dataset (default: 100 rounds)
2. Warmup both instances
3. Run conversations on Instance 1 in original order
4. Run same conversations on Instance 2 in shuffled order
5. Collect and compare performance metrics (TTFT, generation time, etc.) per turn
6. Generate summary statistics and save detailed results to CSV

The random shuffle for Instance 2 helps evaluate cache effectiveness across
different conversation sequences.
"""

import argparse
import asyncio
import json
import logging
import time
import sys
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import openai
import pandas as pd


# ============================================================================
# Logger and Utilities
# ============================================================================

def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] %(levelname)s:{reset} %(message)s "
        + f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name: str, log_level=logging.INFO):
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger


logger = init_logger(__name__, logging.INFO)


# ============================================================================
# Dataset Loader
# ============================================================================

def load_sharegpt_dataset(dataset_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load ShareGPT dataset from local path
    
    Args:
        dataset_path: Path to ShareGPT dataset directory or JSON file
        num_samples: Number of samples to load (None for all)
    
    Returns:
        List of conversation dictionaries
    """
    logger.info(f"Loading ShareGPT dataset from {dataset_path}")
    
    # Check if path is a file or directory
    if os.path.isfile(dataset_path):
        json_file = dataset_path
    elif os.path.isdir(dataset_path):
        # Look for common ShareGPT filenames
        possible_files = [
            os.path.join(dataset_path, "sharegpt.json"),
            os.path.join(dataset_path, "ShareGPT_V3_unfiltered_cleaned_split.json"),
            os.path.join(dataset_path, "sg_90k_part1.json"),
        ]
        json_file = None
        for f in possible_files:
            if os.path.exists(f):
                json_file = f
                break
        
        # If not found, use the first .json file
        if json_file is None:
            json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
            if json_files:
                json_file = os.path.join(dataset_path, json_files[0])
            else:
                raise FileNotFoundError(f"No JSON file found in {dataset_path}")
    else:
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    logger.info(f"Reading dataset file: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract conversations
    conversations = []
    for item in data:
        if isinstance(item, dict) and 'conversations' in item:
            conversations.append(item)
        elif isinstance(item, dict):
            # Some formats might have different structures
            conversations.append(item)
    
    if num_samples is not None:
        conversations = conversations[:num_samples]
    
    logger.info(f"Loaded {len(conversations)} conversations from dataset")
    return conversations


def extract_prompt_from_conversation(conversation: Dict[str, Any]) -> str:
    """
    Extract the first user message from a conversation
    
    Args:
        conversation: Conversation dictionary from ShareGPT
    
    Returns:
        User's prompt string
    """
    if 'conversations' in conversation:
        convs = conversation['conversations']
        for msg in convs:
            if msg.get('from') in ['human', 'user']:
                return msg.get('value', '')
    
    # Fallback for different formats
    if 'messages' in conversation:
        msgs = conversation['messages']
        for msg in msgs:
            if msg.get('role') == 'user':
                return msg.get('content', '')
    
    return "Hello, can you help me?"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TransferConfig:
    """Configuration for performance test"""
    # vLLM instance URLs
    instance1_url: str
    instance2_url: str
    
    # Model name
    model: str
    
    # Dataset path
    dataset_path: str
    
    # Test parameters
    num_rounds: int  # Number of conversation rounds
    max_tokens: int  # Max tokens per generation
    concurrency: int  # Number of concurrent requests
    
    # Output file
    output: str


@dataclass
class RequestResult:
    """Result of a single request in multi-round conversation"""
    round_id: int  # Which conversation round (1-100)
    turn_id: int  # Which turn in the conversation
    prompt: str
    response: str
    ttft: float  # Time to first token
    generation_time: float
    total_time: float
    prompt_tokens: int
    generation_tokens: int
    instance_id: str  # "instance1" or "instance2"


# ============================================================================
# OpenAI API Client
# ============================================================================

class AsyncOpenAIClient:
    """Async OpenAI client wrapper for vLLM"""
    
    def __init__(self, base_url: str, model: str):
        self.client = openai.AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        self.model = model
    
    async def generate(self, messages: List[Dict[str, str]], max_tokens: int) -> tuple:
        """
        Generate response from message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Max tokens to generate
        
        Returns:
            Tuple of (response_text, ttft, generation_time, total_time, prompt_tokens, completion_tokens)
        """
        start_time = time.time()
        first_token_time = None
        last_content_time = None
        response_text = ""
        tokens_in = 0
        tokens_out = 0
        
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
        )
        
        async for chunk in response:
            # Record TTFT on first chunk with choices (most accurate timing)
            if chunk.choices and first_token_time is None:
                first_token_time = time.time()
            
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta.content
            if delta is not None:
                response_text += delta
                # Record time of last content chunk
                last_content_time = time.time()
            
            # Extract usage info if available
            if hasattr(chunk, 'usage') and chunk.usage:
                tokens_in = chunk.usage.prompt_tokens
                tokens_out = chunk.usage.completion_tokens
        
        # Use last content time as end time (more accurate than waiting for usage chunk)
        end_time = last_content_time if last_content_time else time.time()
        
        ttft = first_token_time - start_time if first_token_time else 0
        generation_time = end_time - first_token_time if first_token_time else 0
        total_time = end_time - start_time
        
        return response_text, ttft, generation_time, total_time, tokens_in, tokens_out


# ============================================================================
# Test Executor
# ============================================================================

class KVTransferTester:
    """Main test executor for multi-round conversation performance measurement"""
    
    def __init__(self, config: TransferConfig):
        self.config = config
        self.instance1_client = AsyncOpenAIClient(config.instance1_url, config.model)
        self.instance2_client = AsyncOpenAIClient(config.instance2_url, config.model)
        self.results: List[RequestResult] = []
        self.results_lock = asyncio.Lock()  # Lock for thread-safe results append
        
        # Load dataset conversations
        logger.info(f"Loading {config.num_rounds} conversations from dataset...")
        self.conversations = load_sharegpt_dataset(config.dataset_path, num_samples=config.num_rounds)
        
        # Parse conversations into multi-turn format
        self.parsed_conversations = self._parse_conversations()
        
        logger.info(f"Loaded {len(self.parsed_conversations)} conversations")
        logger.info(f"Concurrency level: {config.concurrency}")
    
    def _parse_conversations(self) -> List[List[Dict[str, str]]]:
        """
        Parse conversations into multi-turn message format
        
        Returns:
            List of conversations, where each conversation is a list of messages
        """
        parsed = []
        for conv in self.conversations:
            messages = []
            if 'conversations' in conv:
                for msg in conv['conversations']:
                    role = 'user' if msg.get('from') in ['human', 'user'] else 'assistant'
                    content = msg.get('value', '')
                    if content:
                        messages.append({'role': role, 'content': content})
            elif 'messages' in conv:
                for msg in conv['messages']:
                    if 'role' in msg and 'content' in msg:
                        messages.append({'role': msg['role'], 'content': msg['content']})
            
            if messages:
                parsed.append(messages)
        
        return parsed
    
    async def _warmup(self):
        """Warmup both instances"""
        logger.info("Warming up instances...")
        warmup_messages = [{"role": "user", "content": "Hello, this is a warmup message."}]
        
        await self.instance1_client.generate(warmup_messages, 50)
        await self.instance2_client.generate(warmup_messages, 50)
        logger.info("Warmup complete")
    
    async def _run_conversation(self, round_id: int, conversation: List[Dict[str, str]], 
                                instance_name: str, client: AsyncOpenAIClient,
                                semaphore: asyncio.Semaphore):
        """
        Run a multi-turn conversation on one instance
        
        Args:
            round_id: The conversation round number
            conversation: List of messages in the conversation
            instance_name: Name of the instance
            client: OpenAI client for the instance
            semaphore: Semaphore to limit concurrency
        """
        async with semaphore:
            message_history = []
            turn_id = 0
            
            for i, message in enumerate(conversation):
                if message['role'] == 'user':
                    turn_id += 1
                    # Add user message to history
                    message_history.append(message)
                    
                    # Generate response
                    response_text, ttft, gen_time, total_time, prompt_tokens, completion_tokens = \
                        await client.generate(message_history, self.config.max_tokens)
                    
                    # Add assistant response to history
                    message_history.append({'role': 'assistant', 'content': response_text})
                    
                    # Record result (thread-safe)
                    result = RequestResult(
                        round_id=round_id,
                        turn_id=turn_id,
                        prompt=message['content'],
                        response=response_text,
                        ttft=ttft,
                        generation_time=gen_time,
                        total_time=total_time,
                        prompt_tokens=prompt_tokens,
                        generation_tokens=completion_tokens,
                        instance_id=instance_name,
                    )
                    async with self.results_lock:
                        self.results.append(result)
                    
                    logger.info(
                        f"  [{instance_name}] Round {round_id}, Turn {turn_id}: "
                        f"TTFT={ttft:.3f}s, Total={total_time:.3f}s"
                    )
                else:
                    # If there's an assistant message in the dataset, add it to history
                    message_history.append(message)
    
    async def run_tests(self):
        """Run all conversation tests"""
        import random
        
        logger.info("=" * 70)
        logger.info("Starting Multi-Round Conversation Tests")
        logger.info("=" * 70)
        logger.info(f"Instance 1: {self.config.instance1_url}")
        logger.info(f"Instance 2: {self.config.instance2_url}")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"Number of conversation rounds: {self.config.num_rounds}")
        logger.info(f"Concurrency: {self.config.concurrency}")
        logger.info("=" * 70)
        
        # Warmup
        await self._warmup()
        
        # Create conversation order for instance1 (original order)
        conversations_instance1 = list(enumerate(self.parsed_conversations, start=1))
        
        # Create conversation order for instance2 (shuffled order)
        conversations_instance2 = list(enumerate(self.parsed_conversations, start=1))
        random.shuffle(conversations_instance2)
        
        # Create semaphores to limit concurrency per instance
        semaphore1 = asyncio.Semaphore(self.config.concurrency)
        semaphore2 = asyncio.Semaphore(self.config.concurrency)
        
        logger.info("\nRunning conversations on Instance 1 (original order) with concurrency...")
        start_time = time.time()
        tasks1 = []
        for round_id, conversation in conversations_instance1:
            if not conversation:
                continue
            task = asyncio.create_task(
                self._run_conversation(round_id, conversation, "instance1", 
                                     self.instance1_client, semaphore1)
            )
            tasks1.append(task)
        
        await asyncio.gather(*tasks1)
        instance1_duration = time.time() - start_time
        logger.info(f"\nInstance 1 completed in {instance1_duration:.2f}s")
        
        logger.info("\n" + "=" * 70)
        logger.info("Running conversations on Instance 2 (shuffled order) with concurrency...")
        start_time = time.time()
        tasks2 = []
        for round_id, conversation in conversations_instance2:
            if not conversation:
                continue
            task = asyncio.create_task(
                self._run_conversation(round_id, conversation, "instance2", 
                                     self.instance2_client, semaphore2)
            )
            tasks2.append(task)
        
        await asyncio.gather(*tasks2)
        instance2_duration = time.time() - start_time
        logger.info(f"\nInstance 2 completed in {instance2_duration:.2f}s")
        
        # Generate summary
        logger.info("\n" + "=" * 70)
        self._print_summary()
        self._save_results()
    
    def _print_summary(self):
        """Print performance summary"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        df = self._results_to_dataframe()
        
        # Split by instance
        df_instance1 = df[df['instance_id'] == 'instance1']
        df_instance2 = df[df['instance_id'] == 'instance2']
        
        logger.info("\n")
        logger.info("=" * 70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 70)
        
        if len(df_instance1) > 0:
            avg_ttft_1 = df_instance1['ttft'].mean()
            avg_total_1 = df_instance1['total_time'].mean()
            avg_gen_time_1 = df_instance1['generation_time'].mean()
            total_rounds_1 = df_instance1['round_id'].nunique()
            total_turns_1 = len(df_instance1)
            
            logger.info(f"\n  Instance 1:")
            logger.info(f"    Total Conversation Rounds: {total_rounds_1}")
            logger.info(f"    Total Turns: {total_turns_1}")
            logger.info(f"    Average TTFT: {avg_ttft_1:.3f}s")
            logger.info(f"    Average Generation Time: {avg_gen_time_1:.3f}s")
            logger.info(f"    Average Total Time: {avg_total_1:.3f}s")
        
        if len(df_instance2) > 0:
            avg_ttft_2 = df_instance2['ttft'].mean()
            avg_total_2 = df_instance2['total_time'].mean()
            avg_gen_time_2 = df_instance2['generation_time'].mean()
            total_rounds_2 = df_instance2['round_id'].nunique()
            total_turns_2 = len(df_instance2)
            
            logger.info(f"\n  Instance 2:")
            logger.info(f"    Total Conversation Rounds: {total_rounds_2}")
            logger.info(f"    Total Turns: {total_turns_2}")
            logger.info(f"    Average TTFT: {avg_ttft_2:.3f}s")
            logger.info(f"    Average Generation Time: {avg_gen_time_2:.3f}s")
            logger.info(f"    Average Total Time: {avg_total_2:.3f}s")
        
        if len(df_instance1) > 0 and len(df_instance2) > 0:
            logger.info(f"\n  Comparison (Instance 2 vs Instance 1):")
            if avg_ttft_1 > 0:
                ttft_ratio = avg_ttft_2 / avg_ttft_1
                logger.info(f"    TTFT Ratio: {ttft_ratio:.2f}x")
            if avg_total_1 > 0:
                total_ratio = avg_total_2 / avg_total_1
                logger.info(f"    Total Time Ratio: {total_ratio:.2f}x")
        
        logger.info("\n" + "=" * 70 + "\n")
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = {
            'round_id': [r.round_id for r in self.results],
            'turn_id': [r.turn_id for r in self.results],
            'instance_id': [r.instance_id for r in self.results],
            'ttft': [r.ttft for r in self.results],
            'generation_time': [r.generation_time for r in self.results],
            'total_time': [r.total_time for r in self.results],
            'prompt_tokens': [r.prompt_tokens for r in self.results],
            'generation_tokens': [r.generation_tokens for r in self.results],
        }
        return pd.DataFrame(data)
    
    def _save_results(self):
        """Save results to CSV"""
        df = self._results_to_dataframe()
        df.to_csv(self.config.output, index=False)
        logger.info(f"Results saved to {self.config.output}")


# ============================================================================
# Main
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-round conversation performance test for vLLM instances"
    )
    
    # Instance configurations
    parser.add_argument(
        "--instance1-url",
        type=str,
        default="http://localhost:8010/v1",
        help="Instance 1 vLLM URL (default: http://localhost:8010/v1)"
    )
    parser.add_argument(
        "--instance2-url",
        type=str,
        default="http://localhost:8011/v1",
        help="Instance 2 vLLM URL (default: http://localhost:8011/v1)"
    )
    
    # Model and dataset parameters
    parser.add_argument(
        "--model",
        type=str,
        default="/root/autodl-tmp/model",
        help="Model path (default: /root/autodl-tmp/model)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/root/autodl-tmp/sharegpt",
        help="ShareGPT dataset path (default: /root/autodl-tmp/sharegpt)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=100,
        help="Number of conversation rounds (default: 100)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per generation (default: 512)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent conversations per instance (default: 1)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="conversation_results.csv",
        help="Output CSV file (default: conversation_results.csv)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main():
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        global logger
        logger = init_logger(__name__, logging.DEBUG)
    
    # Create config
    config = TransferConfig(
        instance1_url=args.instance1_url,
        instance2_url=args.instance2_url,
        model=args.model,
        dataset_path=args.dataset,
        num_rounds=args.num_rounds,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        output=args.output,
    )
    
    # Run tests
    tester = KVTransferTester(config)
    await tester.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
