#!/bin/bash
# Script to build and push brieflow Docker image using Google Cloud Build
# This ensures compatibility with Google Cloud infrastructure

set -euo pipefail

PROJECT_ID="lasagna-199723"
REGION="us-west1"
REPO="brieflow-repo"
IMAGE="brieflow"

echo "Building brieflow Docker image using Cloud Build..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Repository: $REPO"
echo "Image: $IMAGE"
echo ""

# Submit build to Cloud Build
gcloud builds submit \
    --config=cloudbuild.yaml \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    .

echo ""
echo "Build complete!"
echo "Image pushed to: us-west1-docker.pkg.dev/$PROJECT_ID/$REPO/$IMAGE:latest"
