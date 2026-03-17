###############################################################################
# AthleteView AI Platform - Terraform Outputs
###############################################################################

# ---------------------------------------------------------------------------
# EKS
# ---------------------------------------------------------------------------

output "eks_cluster_endpoint" {
  description = "Endpoint URL for the EKS cluster API server."
  value       = aws_eks_cluster.main.endpoint
}

output "eks_cluster_name" {
  description = "Name of the EKS cluster."
  value       = aws_eks_cluster.main.name
}

output "eks_cluster_certificate_authority" {
  description = "Base64-encoded certificate authority data for the EKS cluster."
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster."
  value       = aws_security_group.eks_cluster.id
}

# ---------------------------------------------------------------------------
# RDS (PostgreSQL / TimescaleDB)
# ---------------------------------------------------------------------------

output "rds_cluster_endpoint" {
  description = "Writer endpoint for the Aurora PostgreSQL cluster."
  value       = aws_rds_cluster.main.endpoint
}

output "rds_cluster_reader_endpoint" {
  description = "Reader endpoint for the Aurora PostgreSQL cluster."
  value       = aws_rds_cluster.main.reader_endpoint
}

output "rds_cluster_port" {
  description = "Port on which the RDS cluster accepts connections."
  value       = aws_rds_cluster.main.port
}

output "rds_database_name" {
  description = "Name of the default database in the RDS cluster."
  value       = aws_rds_cluster.main.database_name
}

# ---------------------------------------------------------------------------
# ElastiCache (Redis)
# ---------------------------------------------------------------------------

output "redis_primary_endpoint" {
  description = "Primary endpoint for the Redis replication group."
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "redis_reader_endpoint" {
  description = "Reader endpoint for the Redis replication group."
  value       = aws_elasticache_replication_group.main.reader_endpoint_address
}

output "redis_port" {
  description = "Port on which Redis accepts connections."
  value       = 6379
}

# ---------------------------------------------------------------------------
# MSK (Kafka)
# ---------------------------------------------------------------------------

output "kafka_bootstrap_brokers_tls" {
  description = "TLS bootstrap broker connection string for the MSK cluster."
  value       = aws_msk_cluster.main.bootstrap_brokers_tls
}

output "kafka_zookeeper_connect_string" {
  description = "ZooKeeper connection string for the MSK cluster."
  value       = aws_msk_cluster.main.zookeeper_connect_string
}

# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

output "s3_vod_bucket_name" {
  description = "Name of the S3 bucket used for VOD and clip storage."
  value       = aws_s3_bucket.vod_storage.id
}

output "s3_vod_bucket_arn" {
  description = "ARN of the S3 VOD storage bucket."
  value       = aws_s3_bucket.vod_storage.arn
}

# ---------------------------------------------------------------------------
# CloudFront CDN
# ---------------------------------------------------------------------------

output "cdn_domain_name" {
  description = "CloudFront distribution domain name for HLS delivery."
  value       = aws_cloudfront_distribution.vod.domain_name
}

output "cdn_distribution_id" {
  description = "CloudFront distribution ID."
  value       = aws_cloudfront_distribution.vod.id
}

# ---------------------------------------------------------------------------
# ECR
# ---------------------------------------------------------------------------

output "ecr_repository_urls" {
  description = "Map of ECR repository names to their URLs."
  value       = { for name, repo in aws_ecr_repository.services : name => repo.repository_url }
}

# ---------------------------------------------------------------------------
# VPC
# ---------------------------------------------------------------------------

output "vpc_id" {
  description = "ID of the VPC."
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets."
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets."
  value       = aws_subnet.public[*].id
}
