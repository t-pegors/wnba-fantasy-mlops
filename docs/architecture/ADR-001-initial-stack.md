# ADR 001: Initial MLOps Stack Selection

**Date:** 2026-02-26  
**Status:** Accepted  

## Context
The project requires a robust, automated Machine Learning Operations (MLOps) pipeline to ingest, process, and version daily basketball statistics (WNBA and Unrivaled). To simulate strict compliance and production standards, the architecture must ensure absolute data provenance, secure credential management, and automated reproducibility without manual intervention.

## Decisions

### 1. Version Control & Automation: GitHub & GitHub Actions
We selected GitHub for source code management and GitHub Actions for CI/CD orchestration.
* **Automation:** Actions allow us to define ephemeral, cron-triggered environments (Ubuntu runners) that execute ingestion scripts precisely at scheduled intervals (e.g., daily at 2:00 AM EST).
* **Audit Trail:** Every automated run, success, or failure is logged immutably, ensuring full transparency of pipeline health.

### 2. Credential Security: GitHub Secrets
We explicitly rejected hardcoding AWS credentials or utilizing localized `.env` files in the repository.
* **Security Posture:** Hardcoded keys present a critical vulnerability, exposing cloud infrastructure to unauthorized access. 
* **Ephemeral Injection:** By utilizing GitHub Secrets, AWS Access Keys are securely encrypted at rest. During the Action execution, these keys are injected dynamically into the isolated runner environment and destroyed immediately upon job completion, leaving zero trace in the repository history or logs.

### 3. Data Provenance & Storage: DVC + AWS S3
We selected Data Version Control (DVC) paired with Amazon S3 for data lifecycle management.
* **The Problem:** Git is designed for text-based source code, not large binary files or datasets. Committing data directly degrades repository performance and violates standard engineering practices.
* **The Solution:** DVC acts as a bridge. It uploads the heavy raw data (CSV game logs) to an encrypted S3 bucket. It then generates lightweight cryptographic hash files (`.dvc`) that act as pointers.
* **Data Provenance:** These `.dvc` pointers *are* committed to Git. This creates an immutable link between a specific version of the code and the exact dataset used at that moment in time. If a model exhibits bias or degraded performance in the future, we can roll back the code *and* the data simultaneously to audit the exact state of the pipeline.

## Consequences
* **Positive:** Highly secure, auditable, and reproducible environment. Protects sensitive infrastructure credentials.
* **Negative:** Introduces a learning curve for local development (requiring developers to run `dvc pull/push` alongside `git pull/push`).