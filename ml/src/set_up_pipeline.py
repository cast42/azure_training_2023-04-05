import argparse
import datetime
import os

from azure.ai.ml import Input, MLClient, Output, command, dsl, load_component
from azure.ai.ml.constants import AssetTypes, TimeZone
from azure.ai.ml.entities import (
    Data,
    Environment,
    JobSchedule,
    RecurrencePattern,
    RecurrenceTrigger,
)
from azure.identity import DefaultAzureCredential

parser = argparse.ArgumentParser()
parser.add_argument("--env", dest="env", required=True)
args = parser.parse_args()

env = args.env

assert args.env in [
    "dev",
    "uat",
    "prd",
], "--env parameter must be either dev, uat or prd. Now it is {args.env}"

user = "lodenachtergaele"
# enter details of your AML workspace
subscription_id = "59a62e46-b799-4da2-8314-f56ef5acf82b"
resource_group = "rg-azuremltraining"
workspace = "dummy-workspace"

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# Data setup
web_path = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)

credit_data = Data(
    name=f"{env}_{user}_creditcard_defaults",
    path=web_path,
    type=AssetTypes.URI_FILE,
    description="Dataset for credit card defaults",
    tags={"source_type": "web", "source": "UCI ML Repo"},
)

credit_data = ml_client.data.create_or_update(credit_data)
print(
    f"Dataset with name {credit_data.name} was registered to workspace,"
    f"the dataset version is {credit_data.version}"
)

cpu_compute_target = "aml-cluster"

# Environment setup
custom_env_name = f"{env}_{user}_aml-scikit-learn"
dependencies_dir = "dependencies"
pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.1.0",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace,"
    f"the environment version is {pipeline_job_env.version}"
)

# Define the components
data_prep_src_dir = "components/data_prep"
data_prep_component = command(
    name=f"{env}_{user}_data_prep_credit_defaults",
    display_name="Data preparation for training",
    description="reads a .xl input, split the input to train and test",
    inputs={
        # "data": Input(type="uri_folder"),
        "data": Input(type="uri_file"),
        "test_train_ratio": Input(type="number"),
    },
    outputs=dict(
        train_data=Output(type="uri_folder", mode="rw_mount"),
        test_data=Output(type="uri_folder", mode="rw_mount"),
    ),
    # The source folder of the component
    code=data_prep_src_dir,
    command="""python data_prep.py \
            --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
            --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)
# The train component is defined in a YAML file, so we'll load it here
train_src_dir = "components/train/"
train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))


@dsl.pipeline(
    compute=cpu_compute_target,
    description="E2E data_perp-train pipeline",
)
def credit_defaults_pipeline(
    pipeline_job_data_input,
    pipeline_job_test_train_ratio,
    pipeline_job_learning_rate,
    pipeline_job_registered_model_name,
):
    # using data_prep_function like a python call with its own inputs
    data_prep_job = data_prep_component(
        data=pipeline_job_data_input,
        test_train_ratio=pipeline_job_test_train_ratio,
    )

    # using train_func like a python call with its own inputs
    train_component(
        train_data=data_prep_job.outputs.train_data,
        test_data=data_prep_job.outputs.test_data,
        learning_rate=pipeline_job_learning_rate,
        registered_model_name=pipeline_job_registered_model_name,
    )

    # a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {
        "pipeline_job_train_data": data_prep_job.outputs.train_data,
        "pipeline_job_test_data": data_prep_job.outputs.test_data,
    }


registered_model_name = f"{env}_{user}_credit_defaults_model"

# Let's instantiate the pipeline with the parameters of our choice
pipeline = credit_defaults_pipeline(
    pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
    pipeline_job_test_train_ratio=0.25,
    pipeline_job_learning_rate=0.05,
    pipeline_job_registered_model_name=registered_model_name,
)

if env in ["dev", "uat"]:
    # Perform a dry run of the pipeline
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        # Project's name
        experiment_name=f"{env}_{user}_gha_credit",
    )
if env == "prd":
    # In prd we only schedule
    schedule_name = f"{env}_{user}_credit"

    schedule_start_time = datetime.datetime.utcnow()
    recurrence_trigger = RecurrenceTrigger(
        frequency="month",
        interval=1,
        schedule=RecurrencePattern(month_days=1, hours=1, minutes=0),
        start_time=schedule_start_time,
        time_zone=TimeZone.ROMANCE_STANDARD_TIME,
    )

    job_schedule = JobSchedule(
        name=schedule_name, trigger=recurrence_trigger, create_job=pipeline
    )

    job_schedule = ml_client.schedules.begin_create_or_update(
        schedule=job_schedule
    ).result()
    print(job_schedule)
