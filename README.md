# Wikidata

This repo contains python file to parse the raw wikidata dump to create `json`s with
entities

## Storage Requirements

Working with Wikidata requires significant storage space. Ensure you have sufficient
capacity before proceeding.

### Downloading the Dump File

Wikidata provides entity dumps in various formats. The recommended file for this process
is `latest-all.json.gz`.

1. **Install aria2 for faster downloads**:

   ```sh
   pip install aria2
   ```

2. **Download the file**:

   ```sh
   aria2c --max-connection-per-server=16 https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz
   ```

   As of 18-Dec-2024, the file size for `latest-all.json.gz` is approximately `140.6
GB`.

   Note: Alternatively, you can download the .bz2 version, but .gz is generally faster
   to process.

## Understanding Wikidata Entities

### What Does a Wikidata Entity Look Like?

A typical Wikidata entity is structured as a JSON object. It includes various attributes
like `id`, `type`, `labels`, `descriptions`, `aliases`, and `claims`. Below is an
example of a Wikidata entity:

Example Entity (`Q42` - Douglas Adams)

```json
{
  "id": "Q42",
  "type": "item",
  "labels": {
    "en": {
      "language": "en",
      "value": "Douglas Adams"
    }
  },
  "descriptions": {
    "en": {
      "language": "en",
      "value": "English author and humorist"
    }
  },
  "aliases": {
    "en": [
      {
        "language": "en",
        "value": "Douglas Noel Adams"
      }
    ]
  },
  "claims": {
    "P31": [
      {
        "mainsnak": {
          "snaktype": "value",
          "property": "P31",
          "datavalue": {
            "value": {
              "entity-type": "item",
              "numeric-id": 5,
              "id": "Q5"
            },
            "type": "wikibase-entityid"
          },
          "datatype": "wikibase-item"
        },
        "type": "statement",
        "id": "Q42$F8F97AE9-C88E-41FA-B527-F34A49EB2E59",
        "rank": "normal"
      }
    ]
  },
  "modified": "2024-12-03T18:56:58Z"
}
```

## Processing Wikidata Entities

The provided `run.py` script extracts data from the `latest-all.json.gz` file,
simplifies it, and writes the processed entities into batch JSON files for further use.

### How the Processing Works

1. **Extracting Key Information**: The script filters out essential fields:

   - `id`: The entity's unique identifier (e.g., `Q42`).
   - `type`: The type of the entity (e.g., `item` or `property`).
   - `labels`: A dictionary of multilingual labels (e.g., English).
   - `descriptions`: A dictionary of multilingual descriptions.
   - `aliases`: Alternative names for the entity.
   - `claims`: Simplified representation of the entity's relationships and attributes.

2. **Batching Entities**:

   - Instead of saving each entity as a separate file, entities are grouped into batches
     (e.g., 10,000 entities per batch). This improves performance and reduces the number
     of files.

3. **Parallel Processing**:

   - The script uses multiprocessing to process entities in parallel, significantly
     speeding up the extraction.

4. **Output Format**:
   - Each batch is saved as a JSON file (`batch_0.json`, `batch_1.json`, etc.) in the
     specified output directory.

### Example of Simplified Output

A simplified representation of the example entity `Q42`:

```json
{
  "id": "Q42",
  "type": "item",
  "labels": { "en": "Douglas Adams" },
  "descriptions": {
    "en": "English author and humorist"
  },
  "aliases": { "en": ["Douglas Noel Adams"] },
  "claims": { "P31": [{ "value": "Q5", "qualifiers": {} }] },
  "modified": "2024-12-03T18:56:58Z"
}
```

## Running the Script

### Prerequisites

- Install the required Python packages:
  ```sh
  pip install ijson
  ```
- Ensure you have enough storage space for the output files.

### Usage

Run the script using the following command:

```sh
python run.py --file_path ./data/latest-all.json.gz --output_dir ./data/entities
--num_workers 8 --num_entities_per_batch 10000
```

#### Arguments:

- `--file_path`: Path to the gzipped Wikidata JSON file.
- `--output_dir`: Directory to store the processed batch files.
- `--num_workers`: Number of parallel workers (default: 8).
- `--num_entities_per_batch`: Number of entities per batch file (default: 10,000).

## Notes and Recommendations

1. **Performance Optimization**:

   - Increase `--num_workers` if you have a multi-core CPU.
   - Adjust `--num_entities_per_batch` based on available memory.

2. **Storage Considerations**:

   - Each batch file can be large depending on the number of entities and their
     complexity. Ensure you have enough disk space for both the input file and the
     output files.

3. **Extending Functionality**:
   - You can modify the script to include additional fields or further simplify the
     output depending on your requirements.

## Contributing

Contributions are what make the open source community such an amazing place to be learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure
   code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Authors

- [Taewoon Kim](https://taewoon.kim/)
