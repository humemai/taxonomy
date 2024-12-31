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

   As of 18-Dec-2024, the file size for `latest-all.json.gz` is approximately `141 GB`.

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

The provided `run_entities.py` script extracts data from the `latest-all.json.gz` file,
simplifies it, and writes the processed entities into batch JSON files for further use.
It only extracts English data to save space.

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
  ```bash
  pip install ijson
  ```
- Ensure you have enough storage space for the output files.

### Usage

Run the script using the following command:

```bash
python run_entities.py <FILE_PATH> <OUTPUT_DIR> --num_workers <NUM_WORKERS> --num_entities_per_batch <NUM_ENTITIES_PER_BATCH>
```

#### Arguments:

- `file_path`: Path to the gzipped Wikidata JSON file (e.g., `latest-all.json.gz`).
- `output_dir`: Directory to store the processed batch files (e.g., `entities/`).
- `--num_workers`: Number of parallel worker processes (default: 4).
- `--num_entities_per_batch`: Number of entities per batch file (default: 50,000).
- `--dummy`: Optional flag to run the script in dummy mode for testing. In dummy mode,
  only 987 entities are processed, and 123 entities per batch are used by default.

## Time, Memory, and Storage Considerations

### Time

On my 32 core machine, it took about 1 day and 3 hours to process the entire `latest-all.json.gz` file
with 32 workers and 50,000 entities per batch. It's important to note that the time depends
on the processing power of your machine and the size of the input file. The total processed entities are 113,343,436.

### Memory

The memory is not a major concern since the script processes entities in parallel and
writes them to files as they are processed. This script only consumes a few GB of
memory.

### Storage

The storage requirements depend on the size of the input file and the number of entities
processed per batch. It's important to have sufficient storage space (~400 GB) for the output files.

## Get all the properties as json

Simply run:

```bash
python run_properties.py
```

This will run the SPARQL query to fetch properties from the Wikidata website, retrieve their aliases and descriptions, and save the results to a file called `properties.json`.

`properties.json` will look like:

```json
[
  {
    "property_id": "P6",
    "label": "head of government",
    "aliases": ["leader", "chief of state"],
    "description": "The principal leader of a government, particularly a nation."
  },
  ...
]
```

Each entry in the JSON file contains the following fields:

- `property_id`: The unique identifier for the property (e.g., `P6`).
- `label`: The name or label of the property (e.g., "head of government").
- `aliases`: A list of alternative labels for the property (e.g., ["leader", "chief of state"]).
- `description`: A description of the property (e.g., "The principal leader of a government, particularly a nation.").

## Process entities (`process_entities.py`)

This script processes batches of Wikidata entity JSON files in parallel. It extracts relevant information, counts missing data, and saves the results in various JSON files. The script supports parallel processing using the specified number of processes.

The following JSON files are generated:

- `entityid2label.json`: Maps entity IDs to their English labels.
- `entity_instance_of.json`: Contains the "instance of" (P31) claims for each entity.
- `entity_subclass_of.json`: Contains the "subclass of" (P279) claims for each entity.
- `properties_used.json`: A count of all properties used, ordered by their occurrences.
- `stats.json`: A summary of missing data (e.g., missing labels or properties).

### Arguments:

- `--num_processes`: Number of parallel processes to use (default: 1).
- `--include_qualifiers`: Whether to include qualifiers in the claims (optional).
- `--dummy`: Processes only 10 files (for testing purposes) (optional).

### Output Files:

- **`entityid2label.json`**: Maps entity IDs to English labels.
- **`entity_instance_of.json`**: Contains "instance of" (P31) claims.
- **`entity_subclass_of.json`**: Contains "subclass of" (P279) claims.
- **`properties_used.json`**: A count of properties used, ordered by occurrences.
- **`stats.json`**: Summary of missing data (e.g., missing labels or claims).

This script is designed for efficiency, processing batches of entities in parallel, and saving the results in JSON files. Make sure to set up the required number of processes to speed up the execution for large datasets.

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
