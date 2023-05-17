from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.ai.ml.entities._deployment.deployment_settings import OnlineRequestSettings
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

dotenv_path = f".env.dev"
load_dotenv(dotenv_path=dotenv_path)

subscription_id = os.getenv("AZUREML_SUBSCRIPTION_ID")
resource_group = os.getenv("AZUREML_RESOURCE_GROUP")
workspace = os.getenv("AZUREML_WORSKPACE_NAME")

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

# Define an endpoint name
endpoint_name = "my-endpoint"

# Example way to define a random name
import datetime

# endpoint_name = "cross-encoder-" + datetime.datetime.now().strftime("%m%d%H%M%f")
endpoint_name = "cross-encoder-v1"

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="this is a sample endpoint",
    auth_mode="key"
)

model = Model(path="/home/azureuser/cloudfiles/code/Users/rubenal/mmarco-mMiniLMv2-L12-H384-v1/pytorch_model.bin")
env = Environment(
    conda_file="/home/azureuser/cloudfiles/code/Users/rubenal/cross-encoder.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)


blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="/home/azureuser/cloudfiles/code/Users/rubenal", scoring_script="scoring/cross-encoder.py"
    ),
    instance_type="Standard_F4s_v2",
    instance_count=1,
    request_settings=OnlineRequestSettings(request_timeout_ms=20000)
)

try: 
    ml_client.online_endpoints.get(name=endpoint_name)
except Exception as e:
    print(e)    
    ml_client.online_endpoints.begin_create_or_update(endpoint)
    
        
if ml_client.online_endpoints.get(name=endpoint_name):
    state = ml_client.online_endpoints.get(name=endpoint_name).provisioning_state
    print(state)
    if state == 'Succeeded':
        print(ml_client.online_endpoints.get(name=endpoint_name))
        try:
            ml_client.online_deployments.begin_create_or_update(blue_deployment)
        except Exception as e:
            print(e)

# logs = ml_client.online_deployments.get_logs(
#     name="blue", endpoint_name=endpoint_name, lines=100
# )
# print(logs)

# not all instance types are supported
# you can setup a deployment from one tenant to another
# if you do begin_create_or_update does not update the changes in the yaml requirements file apparently.
# adjust timeout by https://learn.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints?view=azureml-api-2&tabs=cli#http-status-codes
