import boto3
import csv
import os
import subprocess
import click
import json
import time
from loguru import logger
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from huggingface_hub import Repository
from urllib.parse import urlparse
from pymongo import MongoClient, ReturnDocument

client = MongoClient(os.environ["MONGO_URI"])
db_name = "benchmark"
job_collection_name = "finetune"
server_collection_name = "finetune-server"


def update_server_status(server_id, new_status):
    """
    Updates the status of the server in the MongoDB database using a session.
    :param server_id: The unique identifier of the server.
    :param new_status: The new status to be written for the server.
    """
    try:
        with client.start_session() as session:
            servers_collection = client[db_name][server_collection_name]
            update_result = servers_collection.update_one(
                {"server_id": server_id},
                {"$set": {"status": new_status}},
                upsert=True,
                session=session
            )

            if update_result.matched_count > 0:
                logger.info(f"Server status updated: {server_id}, {new_status}")
            else:
                logger.info(f"New server status inserted: {server_id}, {new_status}")

    except Exception as e:
        logger.error(f"Failed to update server status in MongoDB: {e}")
        raise


def poll_for_jobs(server_id, runner_type, interval=30):
    """
    Polls the MongoDB collection every 'interval' seconds for available jobs and yields them for processing.
    :param server_id: Identifier for the server processing the job.
    :param interval: Polling interval in seconds (default: 30 seconds).
    """
    db = client[db_name]
    collection = db[job_collection_name]

    while True:
        update_server_status(server_id, "Fetching jobs")
        job = collection.find_one_and_update(
            {"status": {"$in": ["pending", "error"]}, "runner_type": runner_type, "retry_times": {"$lt": 2}},
            {"$set": {"status": "running"}, "$inc": {"retry_times": 1}},
            return_document=ReturnDocument.AFTER
        )

        if job:
            processing_status = f"Job {job['_id']} is processing: {job['id']}"
            update_server_status(server_id, processing_status)
            yield job
        else:
            logger.info(f"Existing jobs are already claimed by another instance.")

        time.sleep(interval)


def report_job_results(server_name, job, status):
    db = client[db_name]
    collection = db[job_collection_name]
    collection.update_one({'_id': job['_id']}, {'$set': {'status': status}})


def process_job(job, local_dir, server_id, results_dir="clinical_benchmark/results"):
    """
    Processes a given job using the specified parameters and original logic.
    :param job: A dictionary representing the job to be processed.
    :param local_dir: Local directory path for downloaded models and other files.
    :param server_id: Identifier for the server processing the job.
    :param results_dir: Directory for storing results.
    """
    model_name = job['model_name']
    job_id = str(job['_id'])
    revision = job.get('revision', 'main')
    dataset = job['dataset']
    tasks = job['tasks']
    epoch = str(job.get('epoch', 3))
    learning_rate = str(job.get('learning_rate', '3e-4'))
    max_gen_toks = str(job.get('max_gen_toks', '64'))

    local_model_path = os.path.join(local_dir, model_name)

    try:
        if model_name.startswith('s3://'):
            download_model_from_s3(model_name, local_model_path)
            model_name = local_model_path

        command = [
            'bash', os.path.join('./scripts', 'run_sft.sh'),
            model_name, revision, dataset, dataset, epoch, learning_rate, job_id, tasks, max_gen_toks
        ]
        print (command)
        test = subprocess.run(command, check=True)
        return test

    except (BotoCoreError, ClientError, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error processing job {job['id']}: {e}")
        return None

def download_model_from_s3(s3_url, local_path):
    """
    Downloads a model from an S3 URL to a local file path.
    :param s3_url: The S3 URL of the model (e.g., 's3://mybucket/myfolder/mymodel.tar.gz').
    :param local_path: The local file path where the model will be saved.
    """
    try:
        # Parse the S3 URL
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc
        key = parsed_url.path.lstrip('/')

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, key, local_path)
        logger.info(f"Model downloaded successfully from {s3_url} to {local_path}")

    except (BotoCoreError, ClientError) as e:
        logger.error(f"Failed to download model from S3: {e}")
        raise

@click.command()
@click.option('--local_dir', required=True, help="Local directory for the repository and other data")
@click.option('--runner_type', required=True, help="Type of the runner")
@click.option('--server_id', required=True, help="Identifier for the server")
def main(local_dir, runner_type, server_id):
    try:
        for job in poll_for_jobs(server_id, runner_type):
            logger.info(job)
            result = process_job(job, local_dir, server_id)

            new_status = 'successful' if result else 'error'
            report_job_results(server_id, job, new_status)
            # processing_status = f"Job {job['_id']} submitted: {new_status}"
            # update_server_status(server_id, processing_status)
    except Exception as e:
        raise e
        logger.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()
