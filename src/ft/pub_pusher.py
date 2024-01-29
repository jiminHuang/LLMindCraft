from transformers import TrainerCallback
from datetime import datetime
from dataclasses import asdict
import subprocess
from bson.objectid import ObjectId
from pymongo import MongoClient
from huggingface_hub import HfApi
from src.merge_llama_with_lora import apply_lora
import os
import logging

logger = logging.getLogger(__name__)


class PubCallback(TrainerCallback):
    def __init__(self, model_name_or_path, server_id, job_id, tasks, max_gen_toks):
        self._initialized = False
        self.server_id = server_id
        self.model_name_or_path = model_name_or_path
        self.job_id = job_id
        self.tasks = tasks
        self.max_gen_toks = max_gen_toks

    def setup(self, args, state, model, **kwargs):
        if state.is_world_process_zero:
            self._initialized = True
            self.api = HfApi()
            client = MongoClient(os.environ["MONGO_URI"])
            db = client["benchmark"]
            self.job_collection = db["job"]
            self.finetune_collection = db["finetune"]
            self.server_collection = db["finetune-server"]
            self.run_name = args.output_dir.split("/")[-1]

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model, **kwargs)
        run_name = args.output_dir.split("/")[-1]
        self.update_finetune_status({"status": "training"})

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            return
        run_name = args.output_dir.split("/")[-1]
        self.update_finetune_status({"status": "finished"})

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not self._initialized:
            return
        run_name = args.output_dir.split("/")[-1]
        self.update_finetune_status({"status": "running", "args": args.to_dict(), "state": asdict(state), "logs": logs}) 

    def update_server_status(self, status):
        if not self._initialized:
            return
        self.server_collection.update_one({"server_id": self.server_id}, {"$set": {"status": status, "last_modified": datetime.now()}}, upsert=True)

    def update_finetune_status(self, kwargs):
        if not self._initialized:
            return
        kwargs["last_modified"] = datetime.now()
        self.finetune_collection.update_one({"_id": ObjectId(self.job_id)}, {"$set": kwargs}, upsert=True)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f"Pushing checkpoint in {ckpt_dir}. ...")
            run_name = args.output_dir.split("/")[-1]
            merged_path = os.path.join(args.output_dir, "merged")

            subprocess.run(['python', 'src/merge_llama_with_lora.py', '--model_name_or_path', self.model_name_or_path, '--lora_path', artifact_path, '--output_path', merged_path, '--llama'])
            self.api.create_repo(f"YBXL/{run_name}", private=True, exist_ok=True)
            self.api.upload_folder(folder_path=merged_path, repo_id=f"YBXL/{run_name}", repo_type="model") 
            self.api.create_tag(repo_id=f"YBXL/{run_name}", tag=f"{state.global_step}", exist_ok=True)
            self.job_collection.insert_one({
                "id": run_name,
                "model_name": f"YBXL/{run_name}",
                "revision": f"{state.global_step}",
                "tasks": self.tasks,
                "num_shots": 0,
                "retry_times": 0,
                "status": "pending",
                "runner_type": "fast",
                "inference_engine": "hf-causal-vllm",
                "model_prompt": "mellama",
                "batch_size": 100,
                "max_gen_toks": self.max_gen_toks,
            })
            self.update_finetune_status({"status": "Checkpointing {state.global_step}"})
