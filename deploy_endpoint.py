import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ================== KONFIGURASI S3 (SESUAIKAN) ==================
# Pastikan kamu sudah upload manual model.tar.gz ke path di bawah ini sebelum run skrip ini!
BUCKET = "space-titanic-deploy-shafi-267"  
MODEL_S3_KEY = "model/model.tar.gz"   


ENDPOINT_NAME = "space-titanic-endpoint-t5-v6"
REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.2-1"  
# ================================================================

def get_role_arn():
    try:
        return sagemaker.get_execution_role()
    except:
        iam = boto3.client("iam")
        return iam.get_role(RoleName="LabRole")["Role"]["Arn"]

def main():
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    
    role = get_role_arn()
    model_s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"
    
    print(f"Menggunakan Model URI S3: {model_s3_uri}")
    print(f"Menggunakan IAM Role: {role}")
    
   
    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role,
        entry_point="inference.py",  
        source_dir="src",            
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )
    
    print(f"\n[AWS] Mendeploy SageMaker Endpoint '{ENDPOINT_NAME}' (Tunggu 5-10 menit)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )
    
    
    print("\n[AWS] Mengetes Koneksi Real-time Inference...")
    
    
    sample_raw_input = {
        "instances": [
            {
                "PassengerId": "9999_01",
                "HomePlanet": "Europa",
                "CryoSleep": True,
                "Cabin": "C/123/S",
                "Destination": "55 Cancri e",
                "Age": 28.0,
                "VIP": False,
                "RoomService": 0.0,
                "FoodCourt": 0.0,
                "ShoppingMall": 0.0,
                "Spa": 0.0,
                "VRDeck": 0.0,
                "Name": "John Doe"
            }
        ]
    }
    
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    import json
    
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(sample_raw_input)
        )
        print("\nHasil test prediksi dari AWS:")
        print(response["Body"].read().decode("utf-8"))
        print(f"\nEndpoint '{ENDPOINT_NAME}' aktif dan sukses terintegrasi!")
    except Exception as e:
        print(f"\nGagal melakukan invoke endpoint. Cek CloudWatch log. Detail: {str(e)}")

if __name__ == "__main__":
    main()