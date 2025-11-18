"""
Download the dataset for the 2025 FathomNet competition.
"""

import argparse
import asyncio
import csv
import logging
from pathlib import Path

from coco_lib.objectdetection import ObjectDetectionDataset
from httpx import AsyncClient
from PIL import Image
from tqdm import tqdm as tqdm_sync
from tqdm.asyncio import tqdm_asyncio as tqdm_async

LOGGER = logging.getLogger("fathomnet2025")
HANDLER = logging.StreamHandler()
HANDLER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDLER)

async def download_image(client: AsyncClient, url: str, output_path: Path) -> None:
    """
    Downloads an image from a URL to the specified output path.

    Args:
        client (AsyncClient): The HTTP client to use for the download.
        url (str): The URL of the image.
        output_path (Path): The file path to save the image.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    if output_path.exists():  # skip download if the file already exists
        LOGGER.debug(f"{output_path} already exists, skipping download")
        return
    response = await client.get(url)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)
    LOGGER.debug(f"Downloaded {url} to {output_path}")


def crop_and_save_image(image_path: Path, bbox: list[int], output_path: Path) -> None:
    """
    Crops an image to the given bounding box and saves it to the specified path.

    Args:
        image_path (Path): Path to the source image.
        bbox (list[int]): Bounding box in [x, y, width, height] format.
        output_path (Path): Path to save the cropped image.

    Raises:
        IOError: If image reading or writing fails.
    """
    LOGGER.debug(f"Cropping {image_path} {bbox} to {output_path}")
    with Image.open(image_path) as img:
        x, y, w, h = bbox
        cropped = img.crop((x, y, x + w, y + h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
    LOGGER.debug(f"Cropped {image_path} {bbox} to {output_path}")


def write_annotations_csv(output_path: Path, rows: list[list[str]]) -> None:
    """
    Writes the annotations CSV file with given rows.

    Args:
        output_path (Path): Path to the CSV file.
        rows (list[list[str]]): List of rows to write to the file.
    """
    with open(output_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    LOGGER.info(f"Wrote {len(rows)} annotations to {output_path}")


async def process_dataset(
    dataset_path: Path, output_dir: Path, max_concurrent_downloads: int = 10
) -> None:
    """
    Processes a COCO dataset to download images, crop them to ROIs, and save annotations.

    Args:
        dataset_path (Path): Path to the COCO dataset YAML file.
        output_dir (Path): Base directory for the output.
        max_concurrent_downloads (int): Maximum number of concurrent downloads.
    """
    dataset = ObjectDetectionDataset.load(dataset_path)
    images_dir = output_dir / "images"
    rois_dir = output_dir / "rois"
    annotations_csv_path = output_dir / "annotations.csv"

    # set test_bool to True
    if 'test' in dataset_path.name:
        test_bool = True
    else:
        test_bool = False

    semaphore = asyncio.Semaphore(max_concurrent_downloads)

    async def download_image_limited(
        client: AsyncClient, url: str, output_path: Path
    ) -> None:
        """
        Download an image with a semaphore to limit concurrent downloads.

        Args:
            client (AsyncClient): The HTTP client to use for the download.
            url (str): The URL of the image.
            output_path (Path): The file path to save the image.
        """
        async with semaphore:
            await download_image(client, url, output_path)

    image_paths = {
        image.id: images_dir / f"{image.id}.{image.coco_url.split('.')[-1]}"
        for image in dataset.images
    }
    LOGGER.info(f"Downloading {len(dataset.images)} images to {images_dir}")
    async with AsyncClient() as client:
        download_tasks = []
        for image in dataset.images:
            image_path = image_paths[image.id]
            download_tasks.append(
                download_image_limited(client, image.coco_url, image_path)
            )

        for task in tqdm_async.as_completed(
            download_tasks, total=len(download_tasks), desc="Downloading images"
        ):
            await task

    category_names = {category.id: category.name for category in dataset.categories}
    image_map = {image.id: image for image in dataset.images}
    rows = []
    LOGGER.info(f"Processing {len(dataset.annotations)} annotations")
    for annotation in tqdm_sync(dataset.annotations, desc="Processing annotations"):
        image = image_map[annotation.image_id]
        image_path = image_paths[image.id]
        roi_path = rois_dir / f"{annotation.image_id}_{annotation.id}.png"
        crop_and_save_image(image_path, annotation.bbox, roi_path)
        if test_bool:
            rows.append([str(roi_path.resolve()), None])
        else:
            rows.append([str(roi_path.resolve()), category_names[annotation.category_id]])
            
    write_annotations_csv(annotations_csv_path, rows)

    print(f"Saved dataset to {output_dir.resolve().absolute()}")


def main() -> None:
    """
    Main function to parse arguments and process the COCO dataset.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "dataset_path", type=Path, help="Path to the COCO dataset YAML file."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Base directory for the output files."
    )
    parser.add_argument(
        "-n",
        "--num-downloads",
        type=int,
        default=10,
        help="Maximum number of concurrent downloads.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity. Use -vv for debug level.",
    )
    args = parser.parse_args()

    if args.num_downloads <= 0:
        parser.error("Number of downloads must be greater than 0.")

    log_level = (
        logging.DEBUG
        if args.verbose >= 2
        else logging.INFO
        if args.verbose >= 1
        else logging.WARNING
    )
    LOGGER.setLevel(log_level)

    try:
        asyncio.run(
            process_dataset(args.dataset_path, args.output_dir, args.num_downloads)
        )
    except KeyboardInterrupt:
        print("Download interrupted.")


if __name__ == "__main__":
    main()
