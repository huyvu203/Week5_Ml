#!/usr/bin/env python3
"""
Air Quality Prediction Request Script

This script sends prediction requests to the Google Cloud Vertex AI endpoint
for air quality forecasting using the trained AutoML model.

Author: Generated for Week 5 Cloud-Native ML Platform Project
"""

import json
import requests
import os
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth
from datetime import datetime, timedelta
from loguru import logger
import sys

# Configure loguru
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
logger.add("/home/huyvu/Projects/week5/logs/user_requests.log", 
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {function} | {message}", 
           level="DEBUG", rotation="10 MB")

# Configuration - Update these with your actual values
PROJECT_ID = "your-project-id"  # Replace with your Google Cloud project ID
LOCATION = "us-central1"  # Replace with your model location
ENDPOINT_ID = "your-endpoint-id"  # Replace with your deployed endpoint ID

def get_access_token():
    """Get Google Cloud access token for authentication."""
    try:
        # Use Application Default Credentials
        credentials, project = google.auth.default()
        auth_request = Request()
        credentials.refresh(auth_request)
        return credentials.token
    except Exception as e:
        logger.error(f"Failed to get access token: {e}")
        logger.error("Make sure you're authenticated with: gcloud auth application-default login")
        return None

def make_prediction_request(instance_data):
    """
    Send a prediction request to the Vertex AI endpoint.
    
    Args:
        instance_data (dict): The input features for prediction
        
    Returns:
        dict: Prediction response or None if failed
    """
    token = get_access_token()
    if not token:
        return None
    
    # Construct the endpoint URL
    endpoint_url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
    )
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Prepare request payload
    payload = {
        "instances": [instance_data]
    }
    
    try:
        logger.info(f"Sending prediction request for location {instance_data['location_id']}")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(endpoint_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Prediction successful for location {instance_data['location_id']}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def create_example_requests():
    """
    Create example prediction requests using realistic data from the training set.
    
    Returns:
        list: List of example request instances
    """
    # Future timestamps for prediction (24 hours ahead from current time)
    base_time = datetime.now()
    future_times = [
        base_time + timedelta(hours=1),
        base_time + timedelta(hours=6), 
        base_time + timedelta(hours=12)
    ]
    
    # Example instances based on actual locations from the training data
    examples = [
        {
            "location_id": 648,
            "datetimeUtc": future_times[0].strftime("%Y-%m-%dT%H:00:00Z"),
            "latitude": 40.641819,
            "longitude": -74.018707
        },
        {
            "location_id": 3041962,
            "datetimeUtc": future_times[1].strftime("%Y-%m-%dT%H:00:00Z"),
            "latitude": 40.5887458,
            "longitude": -73.9838231
        },
        {
            "location_id": 648,
            "datetimeUtc": future_times[2].strftime("%Y-%m-%dT%H:00:00Z"),
            "latitude": 40.641819,
            "longitude": -74.018707
        }
    ]
    
    return examples

def main():
    """Main function to demonstrate air quality prediction requests."""
    logger.info("Starting air quality prediction requests")
    
    # Ensure logs directory exists
    os.makedirs("/home/huyvu/Projects/week5/logs", exist_ok=True)
    
    # Check configuration
    if PROJECT_ID == "your-project-id" or ENDPOINT_ID == "your-endpoint-id":
        logger.error("Please update PROJECT_ID and ENDPOINT_ID in the script configuration")
        logger.error("You can find these values in your Google Cloud Console")
        return
    
    # Create example requests
    examples = create_example_requests()
    
    logger.info(f"Sending {len(examples)} example prediction requests...")
    
    # Send each request
    for i, example in enumerate(examples, 1):
        logger.info(f"\n--- Example Request {i} ---")
        logger.info(f"Input: {json.dumps(example, indent=2)}")
        
        result = make_prediction_request(example)
        
        if result:
            logger.info(f"Prediction Result: {json.dumps(result, indent=2)}")
            
            # Extract predicted value if available
            if 'predictions' in result and len(result['predictions']) > 0:
                prediction = result['predictions'][0]
                if 'value' in prediction:
                    predicted_value = prediction['value']
                    logger.info(f"Predicted air quality value: {predicted_value}")
                else:
                    logger.info(f"Prediction structure: {prediction}")
        else:
            logger.error(f"Failed to get prediction for example {i}")
        
        logger.info("=" * 50)
    
    logger.info("Air quality prediction requests completed")

def test_single_request():
    """Test function for a single prediction request."""
    logger.info("Testing single prediction request")
    
    # Single test example
    test_instance = {
        "location_id": 648,
        "datetimeUtc": "2025-09-02T12:00:00Z",
        "latitude": 40.641819,
        "longitude": -74.018707
    }
    
    result = make_prediction_request(test_instance)
    
    if result:
        print("\n✅ Test Request Successful!")
        print(f"Input: {json.dumps(test_instance, indent=2)}")
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print("\n❌ Test Request Failed!")
        print("Check the logs for error details")

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Run all example requests
    main()
    
    # Or run a single test request
    # test_single_request()
