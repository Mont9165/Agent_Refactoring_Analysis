#!/bin/bash
set -e

echo "=============================================="
echo "  Agent Refactoring Analysis Environment"
echo "=============================================="
echo ""

# Check if RefactoringMiner is set up
if [ ! -f "tools/RefactoringMiner/build/libs/RM-fat.jar" ]; then
    echo "Setting up RefactoringMiner..."
    echo "This may take a few minutes on first run..."
    echo ""
    
    # Clone and build RefactoringMiner
    mkdir -p tools
    if [ ! -d "tools/RefactoringMiner" ]; then
        echo "Cloning RefactoringMiner..."
        git clone https://github.com/tsantalis/RefactoringMiner.git tools/RefactoringMiner
    fi
    
    cd tools/RefactoringMiner
    echo "Building RefactoringMiner..."
    chmod +x gradlew
    ./gradlew shadowJar
    cd /app
    
    if [ -f "tools/RefactoringMiner/build/libs/RM-fat.jar" ]; then
        echo "✓ RefactoringMiner setup complete!"
    else
        echo "⚠ RefactoringMiner setup may have failed"
    fi
    echo ""
fi

# Copy OAuth properties if not exists
if [ ! -f "github-oauth.properties" ] && [ -f "tools/RefactoringMiner/github-oauth.properties" ]; then
    cp tools/RefactoringMiner/github-oauth.properties .
fi

# Display available commands
echo "Available analysis commands:"
echo ""
echo "1. Extract Java PRs:"
echo "   python scripts/1_simple_java_extraction.py"
echo ""
echo "2. Extract commits:"
echo "   python scripts/2_extract_commits.py"
echo ""
echo "3. Detect refactoring (pattern-based):"
echo "   python scripts/3_detect_refactoring.py"
echo ""
echo "4. RefactoringMiner analysis:"
echo "   python scripts/4_refminer_analysis.py"
echo "   REFMINER_MAX_COMMITS=100 python scripts/4_refminer_analysis.py"
echo ""
echo "Environment ready! Add your GitHub token to github-oauth.properties for full RefactoringMiner functionality."
echo ""

exec "$@"