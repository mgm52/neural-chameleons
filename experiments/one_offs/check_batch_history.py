#!/usr/bin/env python3
"""
Check OpenAI batch API history and status.
"""

import openai
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables (override existing ones)
load_dotenv(override=True)

def check_specific_batch(batch_id):
    """Check a specific batch by ID."""
    try:
        # Debug: Show which API key we're using
        api_key = os.getenv('OPENAI_API_KEY')
        print(f"Using API key: {api_key[:10] + '...' if api_key else 'None'}")
        
        client = openai.OpenAI()
        batch = client.batches.retrieve(batch_id)
        
        created_time = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Created: {created_time}")
        print(f"Endpoint: {batch.endpoint}")
        print(f"Completion Window: {batch.completion_window}")
        
        if hasattr(batch, 'request_counts') and batch.request_counts:
            print(f"Requests - Total: {batch.request_counts.total}, "
                  f"Completed: {batch.request_counts.completed}, "
                  f"Failed: {batch.request_counts.failed}")
        
        if batch.output_file_id:
            print(f"Output File ID: {batch.output_file_id}")
        
        if batch.error_file_id:
            print(f"Error File ID: {batch.error_file_id}")
            
        return batch
        
    except openai.NotFoundError:
        print(f"Batch {batch_id} not found. It may have expired or been created with a different API key.")
        return None
    except Exception as e:
        print(f"Error checking batch {batch_id}: {e}")
        return None

def check_batch_history():
    """Check and display OpenAI batch history."""
    try:
        client = openai.OpenAI()
        
        # Get all batches
        batches = client.batches.list(limit=100)
        
        if not batches.data:
            print("No batches found in your OpenAI account.")
            return
        
        print(f"Found {len(batches.data)} batches:")
        print("-" * 80)
        
        for batch in batches.data:
            created_time = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Batch ID: {batch.id}")
            print(f"Status: {batch.status}")
            print(f"Created: {created_time}")
            print(f"Endpoint: {batch.endpoint}")
            print(f"Completion Window: {batch.completion_window}")
            
            if hasattr(batch, 'request_counts') and batch.request_counts:
                print(f"Requests - Total: {batch.request_counts.total}, "
                      f"Completed: {batch.request_counts.completed}, "
                      f"Failed: {batch.request_counts.failed}")
            
            if batch.output_file_id:
                print(f"Output File ID: {batch.output_file_id}")
            
            if batch.error_file_id:
                print(f"Error File ID: {batch.error_file_id}")
                
            print("-" * 40)
        
        # Show active batches
        active_batches = [b for b in batches.data if b.status in ['validating', 'in_progress', 'finalizing']]
        if active_batches:
            print(f"\n{len(active_batches)} active batches found:")
            for batch in active_batches:
                print(f"- {batch.id}: {batch.status}")
        
    except Exception as e:
        print(f"Error checking batch history: {e}")
        if "OPENAI_API_KEY" not in os.environ:
            print("Make sure OPENAI_API_KEY is set in your environment or .env file")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Check specific batch ID provided as argument
        batch_id = sys.argv[1]
        print(f"Checking specific batch: {batch_id}")
        check_specific_batch(batch_id)
    else:
        # Check all batch history
        check_batch_history()