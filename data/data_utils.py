import os
import re
from pathlib import Path
from typing import Optional, Union, List
from datetime import datetime


def _interactive_file_selection(files: List[Path]) -> Optional[Path]:
    """
    Helper function to interactively select a file from a list.
    
    Args:
        files: List of files sorted by modification time (most recent first)
        
    Returns:
        Selected file path or None if user cancels
    """
    if not files:
        return None
    
    # Show top 5 files
    display_files = files[:5]
    
    print("\nFound matching files (sorted by most recent):")
    print("-" * 80)
    
    for i, file_path in enumerate(display_files, 1):
        # Get file modification time
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        file_size = file_path.stat().st_size
        
        # Format file size
        if file_size < 1024:
            size_str = f"{file_size}B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f}KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f}MB"
        
        print(f"{i:2d}. {file_path.name}")
        print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')} | Size: {size_str}")
        #print(f"    Path: {file_path}")
        print()
    
    if len(files) > 5:
        print(f"... and {len(files) - 5} more files")
        print()
    
    while True:
        try:
            choice = input(f"Select file (1-{len(display_files)}) or 'q' to quit: ").strip().lower()
            
            if choice == 'q':
                return None
            
            file_index = int(choice) - 1
            if 0 <= file_index < len(display_files):
                selected_file = display_files[file_index]
                print(f"Selected: {selected_file}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(display_files)}")
                
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            return None


def find_most_recent_directory(folder_path: Union[str, Path], pattern: str = r"\d{8}_\d{6}", manual_confirm: bool = True) -> Optional[Path]:
    """
    Find the most recent directory in a folder that matches a regex pattern.
    
    Args:
        folder_path: Path to the folder to search in
        pattern: Regex pattern to match directory names against (default: timestamped format YYYYMMDD_HHMMSS)
        manual_confirm: If True, show directories and let user pick one interactively
        
    Returns:
        Path to the most recent matching directory, or None if no matches found
        
    Raises:
        FileNotFoundError: If the folder doesn't exist
        ValueError: If the regex pattern is invalid
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    matching_dirs = []
    
    # Find all directories that match the pattern
    for item in folder.iterdir():
        if item.is_dir() and regex.search(item.name):
            matching_dirs.append(item)
    
    if not matching_dirs:
        return None
    
    # Sort directories by modification time (most recent first)
    matching_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    if manual_confirm and len(matching_dirs) > 1:
        print("\nFound timestamped directories (sorted by most recent):")
        print("-" * 60)
        for i, dir_path in enumerate(matching_dirs[:5], 1):
            mtime = datetime.fromtimestamp(dir_path.stat().st_mtime)
            print(f"{i:2d}. {dir_path.name}")
            print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        while True:
            try:
                choice = input(f"Select directory (1-{min(len(matching_dirs), 5)}) or press Enter for most recent: ").strip()
                
                if choice == "":
                    return matching_dirs[0]
                
                dir_index = int(choice) - 1
                if 0 <= dir_index < min(len(matching_dirs), 5):
                    return matching_dirs[dir_index]
                else:
                    print(f"Please enter a number between 1 and {min(len(matching_dirs), 5)}")
                    
            except ValueError:
                print("Please enter a valid number or press Enter")
            except KeyboardInterrupt:
                print("\nOperation cancelled")
                raise
    else:
        return matching_dirs[0]


def find_most_recent_file(folder_path: Union[str, Path], pattern: str, manual_confirm: bool = True) -> Optional[Path]:
    """
    Find the most recent file in a folder that matches a regex pattern.
    
    Args:
        folder_path: Path to the folder to search in
        pattern: Regex pattern to match filenames against
        manual_confirm: If True, show top 10 files and let user pick one interactively
        
    Returns:
        Path to the most recent matching file, or None if no matches found
        
    Raises:
        FileNotFoundError: If the folder doesn't exist
        ValueError: If the regex pattern is invalid
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")
    
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    
    matching_files = []
    
    # Find all files that match the pattern
    for file_path in folder.iterdir():
        if file_path.is_file() and regex.search(file_path.name):
            matching_files.append(file_path)
    
    if not matching_files:
        return None
    
    # Sort files by modification time (most recent first)
    matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    if manual_confirm:
        return _interactive_file_selection(matching_files)
    else:
        # Return the most recent file
        return matching_files[0]


# Example usage:
if __name__ == "__main__":
    # Find the most recent .txt file automatically
    recent_txt = find_most_recent_file(".", r".*\.txt$")
    print(f"Most recent .txt file: {recent_txt}")
    
    # Find the most recent file starting with "data_" with manual confirmation
    recent_data = find_most_recent_file(".", r"^data_.*", manual_confirm=True)
    print(f"Selected data file: {recent_data}")
    
    # Find Python files interactively
    recent_py = find_most_recent_file(".", r".*\.py$", manual_confirm=True)
    print(f"Selected Python file: {recent_py}")
