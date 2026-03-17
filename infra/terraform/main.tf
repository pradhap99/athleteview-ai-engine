# AthleteView AWS Infrastructure
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster for GPU workloads
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "athleteview-${var.environment}"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    gpu = {
      instance_types = ["g5.2xlarge"]  # NVIDIA A10G GPU
      min_size       = 1
      max_size       = 8
      desired_size   = 2
      ami_type       = "AL2_x86_64_GPU"
    }
    general = {
      instance_types = ["m6i.xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
  }
}

# CloudFront CDN for HLS delivery
resource "aws_cloudfront_distribution" "streaming" {
  enabled = true
  comment = "AthleteView Streaming CDN"

  origin {
    domain_name = aws_s3_bucket.hls_output.bucket_regional_domain_name
    origin_id   = "hls-origin"
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "hls-origin"
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 2
    max_ttl     = 10
  }

  restrictions {
    geo_restriction { restriction_type = "none" }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_s3_bucket" "hls_output" {
  bucket = "athleteview-hls-${var.environment}"
}

# TimescaleDB on RDS
resource "aws_db_instance" "timescaledb" {
  engine               = "postgres"
  engine_version       = "16.1"
  instance_class       = "db.r6g.large"
  allocated_storage    = 100
  db_name              = "athleteview"
  username             = "athleteview"
  password             = var.db_password
  skip_final_snapshot  = var.environment != "production"
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "athleteview-${var.environment}"
  description          = "AthleteView session cache"
  node_type            = "cache.r6g.large"
  num_cache_clusters   = 2
  engine               = "redis"
  engine_version       = "7.0"
}
