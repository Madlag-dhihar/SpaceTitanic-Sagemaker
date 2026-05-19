import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "space-titanic-endpoint-t5-v6")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(raw_features: dict) -> dict:
    runtime = get_runtime_client()
    
    
    payload = {"instances": [raw_features]}
    
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.set_page_config(page_title="Spaceship Titanic Predictor")
st.title("Spaceship Titanic Survival Predictor")

st.markdown(
    "Predicts whether a passenger was **transported** to an alternate dimension "
    "based on personal and flight details."
)


col1, col2 = st.columns(2)
with col1:
    PassengerId = st.text_input("Passenger ID (Format: XXXX_XX)", value="0013_01")
    HomePlanet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars", "Unknown"])
    CryoSleep = st.selectbox("Cryo Sleep?", [True, False])
    Cabin = st.text_input("Cabin Code (Format: Deck/Num/Side)", value="B/0/P")
    Destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e", "Unknown"])
    VIP = st.selectbox("VIP Passenger?", [False, True])
    Age = st.number_input("Passenger Age", min_value=0, max_value=100, value=25)

with col2:
    RoomService = st.number_input("Room Service Spending ($)", min_value=0.0, value=0.0)
    FoodCourt = st.number_input("Food Court Spending ($)", min_value=0.0, value=0.0)
    ShoppingMall = st.number_input("Shopping Mall Spending ($)", min_value=0.0, value=0.0)
    Spa = st.number_input("Spa Spending ($)", min_value=0.0, value=0.0)
    VRDeck = st.number_input("VR Deck Spending ($)", min_value=0.0, value=0.0)
    Name = st.text_input("Passenger Full Name", value="John Doe")

if st.button("Predict", type="primary"):
    
    raw_features = {
        "PassengerId": PassengerId,
        "HomePlanet": HomePlanet,
        "CryoSleep": CryoSleep,
        "Cabin": Cabin,
        "Destination": Destination,
        "Age": float(Age),
        "VIP": VIP,
        "RoomService": float(RoomService),
        "FoodCourt": float(FoodCourt),
        "ShoppingMall": float(ShoppingMall),
        "Spa": float(Spa),
        "VRDeck": float(VRDeck),
        "Name": Name
    }
    
    try:
        result = invoke_endpoint(raw_features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        
        prediction_val = result["prediction"][0]
        
        
        if prediction_val == 1:
            st.success("Result: **TRANSPORTED** ")
        else:
            st.error("Result: **NOT TRANSPORTED** ")