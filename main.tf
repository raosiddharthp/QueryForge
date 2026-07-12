# infra/main.tf — provisions everything described in ADR-002/003/004/005
# and the secondary layer of ADR-006's budget governance. Run `terraform
# plan` and read it before `terraform apply` — this creates billable-capable
# resources, even though every one of them is chosen to fit inside Google
# Cloud's Always-Free tier at demo scale (design doc §8.2).

# ── Firestore (ADR-002) ───────────────────────────────────────────────────
resource "google_firestore_database" "default" {
  project                 = var.project_id
  name                    = "(default)"
  location_id             = var.region
  type                    = "FIRESTORE_NATIVE"
  delete_protection_state = "DELETE_PROTECTION_DISABLED" # flip to ENABLED once this is more than a demo
}

# ── Cloud Storage — corpus files + self-hosted model weights cache ───────
resource "google_storage_bucket" "corpus" {
  name                        = "${var.project_id}-corpus"
  project                     = var.project_id
  location                    = "US"
  uniform_bucket_level_access = true
  force_destroy               = false

  lifecycle_rule {
    condition { age = 90 }
    action { type = "Delete" }
  }
}

# ── Secret Manager (ADR-004) ──────────────────────────────────────────────
resource "google_secret_manager_secret" "gemini_api_key" {
  project   = var.project_id
  secret_id = "gemini-api-key"

  replication {
    auto {}
  }
}
# NOTE: the secret VALUE is not managed here on purpose — set it once via:
#   echo -n "your-ai-studio-key" | gcloud secrets versions add gemini-api-key --data-file=-
# Terraform provisions the secret container, never the key material itself.
# This is a Gemini DEVELOPER API key (aistudio.google.com), not a Vertex AI
# credential — see ADR-004 for why that distinction matters for billing.

# ── Cloud Run (ADR-005) ────────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "queryforge_service" {
  name     = "queryforge-service"
  project  = var.project_id
  location = var.region

  deletion_protection = true

  template {
    scaling {
      min_instance_count = 0 # scale-to-zero — this is the Always Free lever (ADR-005)
      max_instance_count = 3 # capped low on purpose; this is a single-tenant demo
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/cloud-run-source-deploy/queryforge-service:latest"

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi" # sentence-transformers + cross-encoder weights, loaded once per instance
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.queryforge_service.name
  role     = "roles/run.invoker"
  member   = "allUsers" # demo-scoped; see README "Production Path" for the auth-gated alternative
}

# ── IAM — least privilege for the Cloud Run default compute SA ───────────
resource "google_project_iam_member" "compute_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${var.project_number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "compute_datastore_user" {
  project = var.project_id
  role    = "roles/datastore.user" # Firestore access
  member  = "serviceAccount:${var.project_number}-compute@developer.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "compute_storage_viewer" {
  bucket = google_storage_bucket.corpus.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.project_number}-compute@developer.gserviceaccount.com"
}

resource "google_secret_manager_secret_iam_member" "compute_secret_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.project_number}-compute@developer.gserviceaccount.com"
}

# ── ADR-006 (revised): budget governance is NOT primarily enforced here ──
# Google's own billing data lags "at least 24 hours" per their published
# docs, which makes a Pub/Sub-notified, billing-data-driven kill switch
# structurally too slow to enforce a $0.01 cap in real time — by the time
# it would fire, the overspend already happened. The primary enforcement
# is build/service/budget_guard.py: a Firestore-transactional, pre-call
# spend check with zero dependency on GCP's billing pipeline or its lag.
#
# The Cloud Billing Budget below is kept as an independent SECONDARY
# tripwire — defense in depth in case budget_guard.py itself has a bug —
# scoped to this project only, per ADR-006, so it can never reach into the
# three sibling apps sharing the parent billing account.
resource "google_billing_budget" "queryforge_secondary_cap" {
  billing_account = var.billing_account_id
  display_name    = "queryforge-secondary-cap"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = "0"
      nanos         = 10000000 # $0.01
    }
  }

  threshold_rules {
    threshold_percent = 1.0
  }
}
