#!/bin/bash
# =============================================================================
# setup_ec2.sh — Run this ONCE on a fresh Amazon Linux 2023 EC2 instance
# =============================================================================
set -e

echo "=== Installing Docker ==="
sudo yum update -y
sudo yum install -y docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user   # lets ec2-user run docker without sudo

echo "=== Installing Docker Compose plugin ==="
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo "=== Installing AWS CLI ==="
sudo yum install -y awscli

echo "=== Creating app directory ==="
mkdir -p ~/portfolioai/outputs
mkdir -p ~/portfolioai/client_input

echo "=== Creating .env file (fill in your values) ==="
cat > ~/portfolioai/.env << 'EOF'
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_DEFAULT_REGION=us-east-1
TAVILY_API_KEY=your_tavily_api_key_here
EOF

echo "=== Copying docker-compose.yml ==="
# After running this script, copy your docker-compose.yml into ~/portfolioai/
# scp docker-compose.yml ec2-user@<EC2_HOST>:~/portfolioai/

echo ""
echo "============================================================"
echo "Setup complete. Next steps:"
echo "  1. Edit ~/portfolioai/.env with your real credentials"
echo "  2. Copy docker-compose.yml to ~/portfolioai/"
echo "  3. Create an ECR repository named 'portfolioai':"
echo "     aws ecr create-repository --repository-name portfolioai --region us-east-1"
echo "  4. Add these GitHub Secrets to your repo:"
echo "     AWS_ACCESS_KEY_ID"
echo "     AWS_SECRET_ACCESS_KEY"
echo "     AWS_DEFAULT_REGION"
echo "     TAVILY_API_KEY"
echo "     EC2_HOST        <- your EC2 public IP or DNS"
echo "     EC2_USER        <- ec2-user"
echo "     EC2_SSH_KEY     <- paste the full contents of your .pem file"
echo "  5. Push to main — the pipeline will do the rest."
echo "  6. Access the chatbot at: http://<EC2_HOST>:8000"
echo "============================================================"
