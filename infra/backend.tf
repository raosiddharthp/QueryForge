# Remote state in a GCS bucket. Create the bucket manually once, before the
# first `terraform init`, since Terraform can't create the bucket it's about
# to store its own state in:
#
#   gsutil mb -l us-central1 gs://queryforge-prod-tfstate
#   gsutil versioning set on gs://queryforge-prod-tfstate
#
# Rename the bucket to match your project_id.

terraform {
  backend "gcs" {
    bucket = "queryforge-prod-tfstate"
    prefix = "terraform/state"
  }
}
