#!/bin/bash
# Simple script to remove backed-up sessions

echo "Removing backed-up sessions from local storage..."

# Remove 2024-02-06 (birds)
if [ -d "/workspace/fieldwork-data/birds/2024-02-06/environment" ]; then
    echo "Removing birds 2024-02-06..."
    du -sh /workspace/fieldwork-data/birds/2024-02-06/environment
    rm -rf /workspace/fieldwork-data/birds/2024-02-06/environment
    echo "✅ Removed"
fi

# Remove 2024-07-11 (rats)
if [ -d "/workspace/fieldwork-data/rats/2024-07-11/environment" ]; then
    echo "Removing rats 2024-07-11..."
    du -sh /workspace/fieldwork-data/rats/2024-07-11/environment
    rm -rf /workspace/fieldwork-data/rats/2024-07-11/environment
    echo "✅ Removed"
fi

echo "Done! Space freed up."
df -h /workspace/fieldwork-data
