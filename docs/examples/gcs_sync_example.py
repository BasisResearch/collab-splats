"""
Example: Syncing Splatter data with Google Cloud Storage

This example shows how to use the GCS helper functions to:
1. Push Splatter output to GCS after processing
2. Pull existing data from GCS for analysis

Requirements:
    pip install collab-data
"""

from collab_splats.wrapper import Splatter
from collab_splats.utils import push_to_gcs, pull_from_gcs


# Example 1: Run pipeline and push to GCS
def example_push():
    """Run pipeline and upload results to GCS."""
    splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')
    splatter.run_pipeline()

    # Push all results to GCS
    uploaded = push_to_gcs(splatter)
    print(f"Uploaded {uploaded} files")


# Example 2: Pull existing mesh data from GCS
def example_pull_mesh():
    """Download only mesh files from GCS."""
    splatter = Splatter.from_config_file('birds_date-02062024_video-C0043.yaml')

    # Download only mesh files
    downloaded = pull_from_gcs(splatter, include=['mesh'])
    print(f"Downloaded {downloaded} mesh files")

    # Now use the mesh
    splatter.query_mesh(queries=['tree', 'feeder'], output_fn='query_results.ply')


# Example 3: Pull all data from GCS
def example_pull_all():
    """Download all data from GCS for a session."""
    splatter = Splatter.from_config_file('rats_date-07112024_video-C0119.yaml')

    # Download everything
    downloaded = pull_from_gcs(splatter, overwrite=False)
    print(f"Downloaded {downloaded} files")


# Example 4: Custom exclude patterns when pushing
def example_push_custom():
    """Push with custom exclusion patterns."""
    splatter = Splatter.from_config_file('config.yaml')
    splatter.run_pipeline()

    # Exclude checkpoint files and downsampled images
    uploaded = push_to_gcs(
        splatter,
        exclude=['images_4', 'images_8', '.ckpt']
    )
    print(f"Uploaded {uploaded} files")


if __name__ == "__main__":
    # Run the example you want
    example_push()
    # example_pull_mesh()
    # example_pull_all()
    # example_push_custom()
