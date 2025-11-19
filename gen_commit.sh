#!/bin/bash

# Script: generate_commit_description.sh
# Description: Generate commit descriptions for staged files using Ollama and push to GitHub
# Usage: ./generate_commit_description.sh [file1] [file2] ... [fileN]
#        If no files specified, uses all staged changes

set -e  # Exit on error

# Configuration
OUTPUT_FILE="checkin.txt"
OLLAMA_MODEL="llama3.2"
DEBUG=false
AUTO_COMMIT=false  # Set to true to auto-commit and push
TARGET_BRANCH="main"  # Default branch to push to

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Debug print function
debug() {
    if [ "$DEBUG" = true ]; then
        echo -e "${YELLOW}[DEBUG]${NC} $1" >&2
    fi
}

# Error print function
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Success print function
success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

# Info print function
info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    error "Not a git repository. Please run this from within a git repository."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    error "Ollama is not installed or not in your PATH."
    echo "Please install Ollama from: https://ollama.ai" >&2
    exit 1
fi

# Check if Ollama is running by trying to list models
if ! ollama list &> /dev/null; then
    error "Ollama service is not running."
    echo "Please start Ollama with: ollama serve" >&2
    exit 1
fi

# Check if the model exists
if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "Model '$OLLAMA_MODEL' not found. Available models:" >&2
    ollama list >&2
    echo "" >&2
    echo "Pulling model '$OLLAMA_MODEL'..." >&2
    ollama pull "$OLLAMA_MODEL"
fi

# Function to generate commit description for a single file or set of changes
generate_description() {
    local diff_content="$1"
    local file_name="$2"
    
    if [ -z "$diff_content" ]; then
        echo "No changes detected"
        return
    fi
    
    debug "Diff length: ${#diff_content} characters"
    
    # Truncate diff if too long (Ollama has token limits)
    local max_diff_length=8000
    if [ ${#diff_content} -gt $max_diff_length ]; then
        debug "Truncating diff from ${#diff_content} to $max_diff_length characters"
        diff_content="${diff_content:0:$max_diff_length}
... [diff truncated]"
    fi
    
    local prompt="You are an expert at analyzing code changes and writing clear Git commit descriptions.

Analyze the following git diff and provide a concise commit description.
Focus on:
- What changed (the actual modifications)
- Why it matters (the purpose/benefit)
- Any important technical details

Format your response as bullet points. Be specific and concise.
Do NOT include a commit title/subject line.

Git Diff:
\`\`\`
$diff_content
\`\`\`

Provide a clear, structured description:"
    
    debug "Sending request to Ollama for ${file_name:-all changes}..."
    
    # Call Ollama with timeout
    local description
    description=$(timeout 120s ollama run "$OLLAMA_MODEL" "$prompt" 2>/dev/null)
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        error "Ollama request timed out after 120 seconds"
        echo "Error: Request timed out. The diff might be too large."
        return
    elif [ $exit_code -ne 0 ]; then
        error "Ollama request failed with exit code $exit_code"
        debug "Ollama output: $description"
        echo "Error: Failed to generate description."
        return
    fi
    
    if [ -z "$description" ]; then
        error "Ollama returned empty response"
        echo "Error: No description generated."
        return
    fi
    
    echo "$description"
}

# Function to generate commit message from checkin.txt
generate_commit_message() {
    if [ ! -f "$OUTPUT_FILE" ]; then
        error "Checkin file not found: $OUTPUT_FILE"
        return 1
    fi
    
    # Extract the summary/description from checkin.txt
    # Skip header lines and extract meaningful content
    local commit_msg
    commit_msg=$(cat "$OUTPUT_FILE")
    
    echo "$commit_msg"
}

# Function to perform git add, commit, and push
git_operations() {
    local files=("$@")
    
    info "Starting git operations..."
    
    # Stage files
    if [ ${#files[@]} -gt 0 ]; then
        info "Staging specified files..."
        for file in "${files[@]}"; do
            if [ -f "$file" ]; then
                git add "$file"
                success "Staged: $file"
            else
                error "File not found, skipping: $file"
            fi
        done
    else
        info "Staging all changes..."
        git add -A
        success "Staged all changes"
    fi
    
    # Check if there are changes to commit
    if git diff --cached --quiet; then
        error "No changes staged for commit"
        return 1
    fi
    
    # Generate commit message
    info "Creating commit with description from $OUTPUT_FILE..."
    
    if [ ! -f "$OUTPUT_FILE" ]; then
        error "Checkin file not found: $OUTPUT_FILE"
        return 1
    fi
    
    # Commit using the checkin.txt file
    git commit -F "$OUTPUT_FILE"
    success "Commit created successfully"
    
    # Push to remote
    info "Pushing to origin/$TARGET_BRANCH..."
    
    # Check if remote exists
    if ! git remote get-url origin &> /dev/null; then
        error "No 'origin' remote configured"
        echo "Please add a remote with: git remote add origin <URL>" >&2
        return 1
    fi
    
    # Push to branch
    if git push origin "$TARGET_BRANCH"; then
        success "Successfully pushed to origin/$TARGET_BRANCH"
        return 0
    else
        error "Failed to push to origin/$TARGET_BRANCH"
        echo "You may need to pull first or resolve conflicts" >&2
        return 1
    fi
}

# Main script
main() {
    local files=("$@")
    
    # Clear or create output file
    > "$OUTPUT_FILE"
    
    echo "==================================================================" >> "$OUTPUT_FILE"
    echo "COMMIT DESCRIPTION" >> "$OUTPUT_FILE"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
    echo "==================================================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Check if specific files were provided as arguments
    if [ ${#files[@]} -gt 0 ]; then
        debug "Processing ${#files[@]} specified file(s)"
        
        # Process each file individually
        local processed_count=0
        for file in "${files[@]}"; do
            debug "Checking file: $file"
            
            # Check if file exists
            if [ ! -f "$file" ]; then
                error "File not found: $file"
                continue
            fi
            
            # Get diff for this file (staged or unstaged)
            file_diff=$(git diff HEAD -- "$file" 2>/dev/null)
            
            # If no diff from HEAD, try staged changes
            if [ -z "$file_diff" ]; then
                file_diff=$(git diff --cached -- "$file" 2>/dev/null)
            fi
            
            # If still no diff, try unstaged changes
            if [ -z "$file_diff" ]; then
                file_diff=$(git diff -- "$file" 2>/dev/null)
            fi
            
            if [ -z "$file_diff" ]; then
                echo "⚠️  No changes found in: $file" >&2
                continue
            fi
            
            success "Processing: $file"
            
            echo "## Changes in: $file" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            
            # Generate description
            description=$(generate_description "$file_diff" "$file")
            
            if [ -n "$description" ]; then
                echo "$description" >> "$OUTPUT_FILE"
                processed_count=$((processed_count + 1))
            else
                echo "⚠️  Failed to generate description for: $file" >> "$OUTPUT_FILE"
            fi
            
            echo "" >> "$OUTPUT_FILE"
            echo "---" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        done
        
        if [ $processed_count -eq 0 ]; then
            error "No files were successfully processed"
            rm -f "$OUTPUT_FILE"
            exit 1
        fi
        
        success "Processed $processed_count file(s)"
    else
        # No files specified, process all staged changes
        debug "No files specified, checking for staged changes"
        
        git_diff=$(git diff --cached 2>/dev/null)
        
        if [ -z "$git_diff" ]; then
            # No staged changes, try unstaged
            debug "No staged changes, checking unstaged changes"
            git_diff=$(git diff 2>/dev/null)
            
            if [ -z "$git_diff" ]; then
                error "No changes found (neither staged nor unstaged)"
                echo "Stage your changes with: git add <files>" >&2
                rm -f "$OUTPUT_FILE"
                exit 1
            else
                echo "ℹ️  No staged changes found. Using unstaged changes instead." >&2
            fi
        fi
        
        success "Found changes to analyze"
        
        # Get list of changed files
        changed_files=$(git diff --cached --name-only 2>/dev/null)
        if [ -z "$changed_files" ]; then
            changed_files=$(git diff --name-only 2>/dev/null)
        fi
        
        echo "## Changed Files:" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        while IFS= read -r file; do
            if [ -n "$file" ]; then
                echo "- $file" >> "$OUTPUT_FILE"
            fi
        done <<< "$changed_files"
        echo "" >> "$OUTPUT_FILE"
        
        echo "## Summary of Changes:" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        # Generate description for all changes
        description=$(generate_description "$git_diff" "all changes")
        
        if [ -n "$description" ]; then
            echo "$description" >> "$OUTPUT_FILE"
        else
            error "Failed to generate description"
            echo "Error: Could not generate description" >> "$OUTPUT_FILE"
        fi
        
        echo "" >> "$OUTPUT_FILE"
    fi
    
    echo "==================================================================" >> "$OUTPUT_FILE"
    
    # Display result
    echo "" >&2
    success "Commit description saved to: $OUTPUT_FILE"
    echo "" >&2
    echo "Preview:" >&2
    echo "---" >&2
    cat "$OUTPUT_FILE" >&2
    echo "---" >&2
    
    # Perform git operations if AUTO_COMMIT is true
    if [ "$AUTO_COMMIT" = true ]; then
        echo "" >&2
        git_operations "${files[@]}"
    else
        echo "" >&2
        info "To commit and push these changes, run with -c flag:"
        echo "  $0 -c ${files[*]}" >&2
        echo "" >&2
        info "Or manually run:"
        echo "  git add ${files[*]:-<files>}" >&2
        echo "  git commit -F $OUTPUT_FILE" >&2
        echo "  git push origin $TARGET_BRANCH" >&2
    fi
}

# Parse command line options
while getopts "dcb:h" opt; do
    case $opt in
        d)
            DEBUG=true
            ;;
        c)
            AUTO_COMMIT=true
            ;;
        b)
            TARGET_BRANCH="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [-d] [-c] [-b branch] [file1] [file2] ... [fileN]"
            echo ""
            echo "Options:"
            echo "  -d           Enable debug mode"
            echo "  -c           Auto-commit and push after generating description"
            echo "  -b branch    Specify target branch (default: main)"
            echo "  -h           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Process all staged/unstaged changes"
            echo "  $0 file.py                  # Process specific file"
            echo "  $0 -c file1.py file2.py     # Process, commit, and push files"
            echo "  $0 -c -b develop file.py    # Commit and push to 'develop' branch"
            exit 0
            ;;
        \?)
            error "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

shift $((OPTIND-1))

# Run main function with remaining arguments
main "$@"