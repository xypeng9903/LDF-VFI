import datasets
from pathlib import Path


def to_hf_dataset(args):              
    def gen():
        for path in Path(args.source).rglob("*"):
            if path.is_file() and path.suffix.lower() in args.suffix:
                yield {"video": str(path)}
            
    raw_data = datasets.Dataset.from_generator(gen, cache_dir=Path(args.dest) / ".cache")
    raw_data.save_to_disk(args.dest)
        
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src")
    parser.add_argument("--dst")
    args = parser.parse_args()
    to_hf_dataset(args)