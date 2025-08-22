# ScAdver: Git Commands for GitHub Push

## Step-by-step commands to push ScAdver to GitHub

```bash
# 1. Navigate to the ScAdver directory
cd /Users/shivaprasad/Documents/PROJECTS/GitHub/ScAdver

# 2. Initialize git repository (if not already done)
git init

# 3. Set up remote origin (replace if exists)
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/shivaprasad-patil/ScAdver.git

# 4. Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
build/
dist/
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/

# Environments
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# Model files (optional)
# *.pth
# *.pkl
EOF

# 5. Check git status
git status

# 6. Pull any existing content from GitHub (like LICENSE)
git pull origin main --allow-unrelated-histories

# 7. Add all files
git add .

# 8. Commit changes
git commit -m "ScAdver v1.0.0: Adversarial batch correction for single-cell data

- Renamed from AdverBatchBio to ScAdver
- Complete adversarial training implementation
- Reference-query batch correction support
- MPS/CUDA/CPU device compatibility
- Comprehensive documentation and examples
- Full test coverage with synthetic data validation"

# 9. Push to GitHub
git push -u origin main

# 10. Create release tag (optional)
git tag -a v1.0.0 -m "ScAdver v1.0.0 - Initial release"
git push origin v1.0.0
```

## Authentication Options

### Option A: Personal Access Token
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate token with 'repo' permissions
3. Use your GitHub username and token as password

### Option B: SSH Key
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your-email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys
3. Change remote: `git remote set-url origin git@github.com:shivaprasad-patil/ScAdver.git`

## Verification
After pushing, your repository should contain:
- scadver/ - Main package
- examples/ - Example scripts  
- README.md - Documentation
- setup.py - Package configuration
- LICENSE - Apache 2.0 license
- test_scadver_complete.py - Test script
