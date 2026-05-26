#!/bin/bash
# =============================================================================
# setup_ec2.sh — Run ONCE on a fresh Amazon Linux 2023 EC2 instance
# Use this only if self-hosting on EC2 instead of Railway.
# For Railway deployment, push to GitHub — CI/CD handles everything.
# =============================================================================
set -e

echo "=== Installing Docker ==="
sudo yum update -y
sudo yum install -y docker curl
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user   # lets ec2-user run docker without sudo

echo "=== Installing Docker Compose plugin ==="
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

echo "=== Creating app directories ==="
mkdir -p ~/portfolioai/outputs
mkdir -p ~/portfolioai/client_input

echo "=== Creating .env file (fill in your real values) ==="
cat > ~/portfolioai/.env << 'EOF'
# ── Required ────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# ── No AWS credentials needed ────────────────────────────────────────────
# App uses Anthropic Claude API directly (not via Bedrock)
# Embeddings use HuggingFace (no API key required)
EOF

echo ""
echo "============================================================"
echo "Setup complete. Next steps:"
echo ""
echo "  1. Edit ~/portfolioai/.env with your real API keys:"
echo "     nano ~/portfolioai/.env"
echo ""
echo "  2. Copy your project files to the instance:"
echo "     scp -r . ec2-user@<EC2_HOST>:~/portfolioai/"
echo ""
echo "  3. Start the app:"
echo "     cd ~/portfolioai"
echo "     docker compose up -d --build"
echo ""
echo "  4. Access at: http://<EC2_HOST>:8080"
echo ""
echo "  5. View logs:"
echo "     docker compose logs -f"
echo "============================================================"
