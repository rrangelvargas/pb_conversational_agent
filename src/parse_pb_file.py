import csv
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

def parse_pb_file(file_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Parse a single Protocol Buffer file using the pabulib recommended approach.
    
    Args:
        file_path: Path to the .pb file
        
    Returns:
        Tuple of (meta, projects, votes) dictionaries
    """
    meta = {}
    projects = {}
    votes = {}
    section = ""
    header = []
    
    try:
        with open(file_path, 'r', newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                if str(row[0]).strip().lower() in ["meta", "projects", "votes"]:
                    section = str(row[0]).strip().lower()
                    header = next(reader)
                elif section == "meta":
                    if len(row) >= 2:
                        meta[row[0]] = row[1].strip()
                elif section == "projects":
                    if len(row) >= 2:
                        projects[row[0]] = {}
                        for it, key in enumerate(header[1:]):
                            if it + 1 < len(row):
                                # Don't truncate at comma, preserve full description
                                projects[row[0]][key.strip()] = row[it+1].strip()
                elif section == "votes":
                    if len(row) >= 2:
                        votes[row[0]] = {}
                        for it, key in enumerate(header[1:]):
                            if it + 1 < len(row):
                                votes[row[0]][key.strip()] = row[it+1].strip()
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}, {}, {}
    
    return meta, projects, votes

def save_metadata_to_csv(meta: Dict, file_path: str):
    """Save metadata to CSV file."""
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['key', 'value'])
        for key, value in meta.items():
            writer.writerow([key, value])

def save_metadata_combined_to_csv(all_meta: List[Dict], file_names: List[str], file_path: str):
    """
    Save combined metadata to CSV with one row per file and columns for each metadata key.
    
    Args:
        all_meta: List of metadata dictionaries
        file_names: List of corresponding file names
        file_path: Output file path
    """
    if not all_meta:
        return
    
    # Get all unique metadata keys across all files
    all_keys = set()
    for meta in all_meta:
        all_keys.update(meta.keys())
    
    # Sort keys for consistent column order
    sorted_keys = sorted(all_keys)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Header: filename + all metadata keys
        writer.writerow(['filename'] + sorted_keys)
        
        # One row per file
        for i, meta in enumerate(all_meta):
            file_name = Path(file_names[i]).stem
            row = [file_name]
            for key in sorted_keys:
                row.append(meta.get(key, ''))  # Empty string if key doesn't exist
            writer.writerow(row)

def save_projects_to_csv(projects: Dict, file_path: str):
    """Save projects to CSV file."""
    if not projects:
        return
    
    # Get all unique keys from all projects
    all_keys = set()
    for project_data in projects.values():
        all_keys.update(project_data.keys())
    
    # Sort keys for consistent column order
    sorted_keys = sorted(all_keys)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['project_id'] + sorted_keys)
        writer.writeheader()
        
        for project_id, project_data in projects.items():
            row = {'project_id': project_id}
            for key in sorted_keys:
                row[key] = project_data.get(key, '')
            writer.writerow(row)

def save_votes_to_csv(votes: Dict, file_path: str):
    """Save votes to CSV file."""
    if not votes:
        return
    
    # Get all unique keys from all votes
    all_keys = set()
    for vote_data in votes.values():
        all_keys.update(vote_data.keys())
    
    # Sort keys for consistent column order
    sorted_keys = sorted(all_keys)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['voter_id'] + sorted_keys)
        writer.writeheader()
        
        for voter_id, vote_data in votes.items():
            row = {'voter_id': voter_id}
            for key in sorted_keys:
                row[key] = vote_data.get(key, '')
            writer.writerow(row)

def combine_data(data_list: List[Tuple[Dict, Dict, Dict]], file_names: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    Combine data from multiple files, handling duplicate project IDs intelligently.
    
    Args:
        data_list: List of (meta, projects, votes) tuples
        file_names: List of corresponding file names
        
    Returns:
        Combined (meta, projects, votes) tuple
    """
    combined_meta = {}
    combined_projects = {}
    combined_votes = {}
    
    for i, (meta, projects, votes) in enumerate(data_list):
        file_name = Path(file_names[i]).stem
        
        # Combine metadata (keep unique keys, add file prefix for duplicates)
        for key, value in meta.items():
            if key in combined_meta:
                combined_meta[f"{key}_{file_name}"] = value
            else:
                combined_meta[key] = value
        
        # Combine projects (don't duplicate if same ID, just add source file info)
        for project_id, project_data in projects.items():
            if project_id in combined_projects:
                # Project already exists, add source file info
                if 'source_files' not in combined_projects[project_id]:
                    combined_projects[project_id]['source_files'] = []
                combined_projects[project_id]['source_files'].append(file_name)
            else:
                # New project, add it with source file info
                combined_projects[project_id] = project_data.copy()
                combined_projects[project_id]['source_files'] = [file_name]
        
        # Combine votes (no need to prefix voter IDs since they're unique across files)
        for voter_id, vote_data in votes.items():
            combined_votes[voter_id] = vote_data.copy()
            combined_votes[voter_id]['source_file'] = file_name
    
    return combined_meta, combined_projects, combined_votes

def parse_single_file(input_file: str, output_dir: str):
    """Parse a single PB file and create CSV files."""
    print(f"Parsing single file: {input_file}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse the file
    meta, projects, votes = parse_pb_file(input_file)
    
    if not meta and not projects and not votes:
        print(f"Failed to parse {input_file}")
        return
    
    print(f"Parsing complete!")
    print(f"Metadata: {len(meta)} entries")
    print(f"Projects: {len(projects)} entries")
    print(f"Votes: {len(votes)} entries")
    
    # Save to CSV files
    print("Saving CSV files...")
    
    metadata_file = output_path / 'metadata.csv'
    save_metadata_to_csv(meta, metadata_file)
    print(f"Saved metadata to {metadata_file}")
    
    if projects:
        projects_file = output_path / 'projects.csv'
        save_projects_to_csv(projects, projects_file)
        print(f"Saved {len(projects)} projects to {projects_file}")
    
    if votes:
        votes_file = output_path / 'votes.csv'
        save_votes_to_csv(votes, votes_file)
        print(f"Saved {len(votes)} votes to {votes_file}")

def parse_multiple_files(pattern: str, output_dir: str):
    """Parse multiple PB files matching a pattern and combine the data."""
    # Add default data/raw path if not specified
    if not pattern.startswith('data/raw/'):
        pattern = f"data/raw/*{pattern}*.pb"
    
    print(f"Parsing files matching pattern: {pattern}")
    
    # Find matching files
    matching_files = glob.glob(pattern)
    if not matching_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(matching_files)} matching files:")
    for file in matching_files:
        print(f"  - {file}")
    
    # Create output directory based on the pattern string
    pattern_name = pattern.replace('data/raw/*', '').replace('*.pb', '').strip('*')
    if not pattern_name:
        pattern_name = 'combined'
    
    output_path = Path(output_dir) / pattern_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Parse all files
    all_data = []
    file_names = []
    
    for file_path in matching_files:
        print(f"\nParsing {file_path}...")
        meta, projects, votes = parse_pb_file(file_path)
        all_data.append((meta, projects, votes))
        file_names.append(file_path)
        
        print(f"  Metadata: {len(meta)} entries")
        print(f"  Projects: {len(projects)} entries")
        print(f"  Votes: {len(votes)} entries")
    
    # Combine data
    print(f"\nCombining data from {len(matching_files)} files...")
    combined_meta, combined_projects, combined_votes = combine_data(all_data, file_names)
    
    print(f"Combined results:")
    print(f"  Total metadata: {len(combined_meta)} entries")
    print(f"  Total projects: {len(combined_projects)} entries")
    print(f"  Total votes: {len(combined_votes)} entries")
    
    # Save combined CSV files
    print("Saving combined CSV files...")
    
    # Save metadata in new format (one row per file)
    metadata_file = output_path / 'metadata.csv'
    save_metadata_combined_to_csv([data[0] for data in all_data], file_names, metadata_file)
    print(f"Saved combined metadata to {metadata_file}")
    
    if combined_projects:
        projects_file = output_path / 'projects.csv'
        save_projects_to_csv(combined_projects, projects_file)
        print(f"Saved {len(combined_projects)} combined projects to {projects_file}")
    
    if combined_votes:
        votes_file = output_path / 'votes.csv'
        save_votes_to_csv(combined_votes, votes_file)
        print(f"Saved {len(combined_votes)} combined votes to {votes_file}")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Parse Protocol Buffer (.pb) files and convert to CSV')
    parser.add_argument('--file', '-f', type=str, help='Single .pb file to parse')
    parser.add_argument('--pattern', '-p', type=str, help='Pattern to match multiple files (e.g., "warszawa" for data/raw/*warszawa*.pb)')
    parser.add_argument('--output', '-o', type=str, default='data/parsed', help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    if args.file:
        # Parse single file
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found")
            return
        
        # Create output directory based on file name
        file_name = Path(args.file).stem
        output_dir = Path(args.output) / file_name
        parse_single_file(args.file, output_dir)
        
    elif args.pattern:
        # Parse multiple files matching pattern
        parse_multiple_files(args.pattern, args.output)
        
    else:
        # Default: parse the main file
        default_file = 'data/raw/poland_warszawa_2023_.pb'
        if os.path.exists(default_file):
            output_dir = Path(args.output) / 'poland_warszawa_2023'
            parse_single_file(default_file, output_dir)
        else:
            print("No file specified and default file not found.")
            print("Use --file to parse a single file or --pattern to parse multiple files.")
            print("Example:")
            print("  python parse_pb_file.py --file data/raw/example.pb")
            print("  python parse_pb_file.py --pattern warszawa")

if __name__ == "__main__":
    main() 