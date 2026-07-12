variable "project_id" {
  description = "GCP project ID — rename before apply, must be globally unique. This project is linked to but NOT merged with the shared $10 billing account (ADR-006)."
  type        = string
  default     = "queryforge-prod"
}

variable "region" {
  description = "Primary region for all resources"
  type        = string
  default     = "us-central1"
}

variable "project_number" {
  description = "GCP project number (numeric) — get via: gcloud projects describe PROJECT_ID --format='value(projectNumber)'"
  type        = string
  # No default on purpose — per-project, must be set explicitly in a
  # terraform.tfvars file you do NOT commit.
}

variable "billing_account_id" {
  description = "The shared GCP billing account this project is linked to. The Cloud Billing Budget below is scoped to THIS project only — see ADR-006 on why an account-level budget was rejected."
  type        = string
}
